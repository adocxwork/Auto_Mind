import os
import pickle
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, abort
import google.generativeai as genai
from markdown import markdown
import re

# Import configuration
try:
    from config import AK, APP_HOST, APP_PORT
except ImportError:
    # Fallback to environment variables if config.py doesn't exist
    AK = os.getenv('AK')
    APP_HOST = '0.0.0.0'
    APP_PORT = 5000

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            user_type TEXT NOT NULL CHECK(user_type IN ('customer','mechanic','car_dealer')),
            upi_id TEXT
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id INTEGER NOT NULL,
            receiver_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(sender_id) REFERENCES users(id),
            FOREIGN KEY(receiver_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()

# Initialize database once at startup (compatible with Flask 3.x)
init_db()

# Load the trained model
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the cleaned dataset to get unique values for dropdowns
try:
    df = pd.read_csv('Cleaned_Car.csv')
    print("Dataset loaded successfully!")
    
    # Get unique values for dropdowns
    companies = sorted(df['company'].unique().tolist())
    fuel_types = sorted(df['fuel_type'].unique().tolist())
    years = sorted(df['year'].unique().tolist())
    
    # Get models for each company
    company_models = {}
    for company in companies:
        company_models[company] = sorted(df[df['company'] == company]['name'].unique().tolist())
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    companies = []
    fuel_types = []
    years = []
    company_models = {}

# Configure AI API (non-fatal if it fails)
ai_model = None
if AK and AK != "your_ai_api_key_here":
    try:
        genai.configure(api_key=AK)
        # Use a safe default model; if unavailable, fall back cleanly
        try:
            ai_model = genai.GenerativeModel('gemini-pro')
        except Exception:
            ai_model = genai.GenerativeModel('gemini-pro-latest')
        print("AI API configured successfully!")
    except Exception as e:
        print(f"AI init failed, continuing without chatbot: {e}")
else:
    print("AK not configured. Chatbot functionality will be limited.")
    print("To enable chatbot: Edit config.py and add your AI API key")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html', 
                         companies=companies, 
                         fuel_types=fuel_types, 
                         years=years)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# -------------------- Authentication & Profiles --------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user_type = request.form.get('user_type', '').strip()
        if not username or not password or user_type not in ['customer','mechanic','car_dealer']:
            return render_template('register.html', error='All fields are required.', username=username)
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('INSERT INTO users (username, password, user_type) VALUES (?,?,?)',
                        (username, password, user_type))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template('register.html', error='Username already exists.', username=username)
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, username, password, user_type, upi_id FROM users WHERE username = ?', (username,))
        user = cur.fetchone()
        conn.close()
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_type'] = user['user_type']
            return redirect(url_for('home'))
        return render_template('login.html', error='Invalid credentials.', username=username)
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    cur = conn.cursor()
    if request.method == 'POST':
        upi_id = request.form.get('upi_id', '').strip()
        cur.execute('UPDATE users SET upi_id = ? WHERE id = ?', (upi_id, session['user_id']))
        conn.commit()
    cur.execute('SELECT id, username, user_type, upi_id FROM users WHERE id = ?', (session['user_id'],))
    user = cur.fetchone()
    conn.close()
    return render_template('profile.html', user=user)

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user_id = session['user_id']
    conn = get_db_connection()
    cur = conn.cursor()
    # Delete messages related to the user, then the user
    cur.execute('DELETE FROM messages WHERE sender_id = ? OR receiver_id = ?', (user_id, user_id))
    cur.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    session.clear()
    return redirect(url_for('home'))

# -------------------- User Listings --------------------

@app.route('/mechanics')
def mechanics():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, user_type, IFNULL(upi_id,'') as upi_id FROM users WHERE user_type='mechanic' ORDER BY username")
    users = cur.fetchall()
    conn.close()
    return render_template('users.html', title='Mechanics', users=users)

@app.route('/dealers')
def dealers():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, user_type, IFNULL(upi_id,'') as upi_id FROM users WHERE user_type='car_dealer' ORDER BY username")
    users = cur.fetchall()
    conn.close()
    return render_template('users.html', title='Car Dealers', users=users)

# -------------------- Simple User-to-User Chat --------------------

def require_login():
    if 'user_id' not in session:
        return False
    return True

@app.route('/chat_users')
def chat_users():
    if not require_login():
        return redirect(url_for('login'))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, user_type FROM users WHERE id != ? ORDER BY username', (session['user_id'],))
    users = cur.fetchall()
    conn.close()
    return render_template('chat_users.html', users=users)

@app.route('/chat_with/<int:other_id>', methods=['GET', 'POST'])
def chat_with(other_id):
    if not require_login():
        return redirect(url_for('login'))
    conn = get_db_connection()
    cur = conn.cursor()
    # Load all other users for sidebar
    cur.execute('SELECT id, username, user_type FROM users WHERE id != ? ORDER BY username', (session['user_id'],))
    sidebar_users = cur.fetchall()
    # Validate other user exists
    cur.execute('SELECT id, username FROM users WHERE id = ?', (other_id,))
    other = cur.fetchone()
    if not other:
        conn.close()
        abort(404)
    if request.method == 'POST':
        message = request.form.get('message', '').strip()
        if message:
            cur.execute(
                'INSERT INTO messages (sender_id, receiver_id, message, created_at) VALUES (?,?,?,?)',
                (session['user_id'], other_id, message, datetime.utcnow().isoformat())
            )
            conn.commit()
    # Load history (both directions)
    cur.execute(
        """
        SELECT m.id, m.sender_id, m.receiver_id, m.message, m.created_at,
               su.username AS sender_name, ru.username AS receiver_name
        FROM messages m
        JOIN users su ON su.id = m.sender_id
        JOIN users ru ON ru.id = m.receiver_id
        WHERE (m.sender_id = ? AND m.receiver_id = ?) OR (m.sender_id = ? AND m.receiver_id = ?)
        ORDER BY m.id ASC
        """,
        (session['user_id'], other_id, other_id, session['user_id'])
    )
    messages = cur.fetchall()
    conn.close()
    return render_template('chat.html', other=other, messages=messages, users=sidebar_users)

@app.route('/get_models/<company>')
def get_models(company):
    """Get models for a specific company"""
    if company in company_models:
        return jsonify(company_models[company])
    return jsonify([])

@app.route('/predict', methods=['POST'])
def predict():
    """Handle car price prediction"""
    try:
        if model is None:
            return jsonify({'success': False, 'error': 'Model not loaded'})
        
        # Get form data
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        kms_driven = int(request.form.get('kms_driven'))
        fuel_type = request.form.get('fuel_type')
        
        # Validate inputs
        if not all([company, car_model, year, kms_driven, fuel_type]):
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        # Create prediction input
        prediction_data = pd.DataFrame([[
            car_model,
            company,
            year,
            kms_driven,
            fuel_type
        ]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Make prediction
        predicted_price = model.predict(prediction_data)[0]
        
        # Format the price
        formatted_price = f"₹{predicted_price:,.0f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot messages"""
    try:
        if ai_model is None:
            return jsonify({
                'success': False, 
                'response': 'Chatbot service is currently unavailable. Please set the AK environment variable.'
            })
        
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'success': False, 'response': 'Please provide a message.'})
        
        # Create a context-aware prompt for car-related queries
        prompt = f"""
        You are an AI assistant for Auto_Mind, a car price prediction platform. 
        You help users with car-related questions, pricing information, vehicle specifications, 
        buying and selling advice, car maintenance tips, and general automotive knowledge.
        
        User's question: {message}
        
        Please provide a helpful, accurate, and detailed response. If the question is about car pricing,
        you can mention that Auto_Mind provides AI-powered price predictions. Keep your response 
        conversational and informative.
        """
        
        # Generate response using AI
        response = ai_model.generate_content(prompt)
        
        # Convert markdown to HTML
        html_response = markdown(response.text)
        
        return jsonify({
            'success': True,
            'response': html_response
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'success': False, 
            'response': 'Sorry, I encountered an error. Please try again later.'
        })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', APP_PORT))
    app.run(debug=False, host='0.0.0.0', port=port)
