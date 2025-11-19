import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import google.generativeai as genai
from markdown import markdown
import re
from datetime import datetime

# Import configuration
try:
    from config import AK, APP_HOST, APP_PORT
except ImportError:
    # Fallback to environment variables if config.py doesn't exist
    AK = os.getenv('AK')
    APP_HOST = '0.0.0.0'
    APP_PORT = 5000

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')

DB_PATH = os.path.join(os.path.dirname(__file__), 'app.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                account_type TEXT NOT NULL CHECK(account_type IN ('Customer','Mechanic','Car Dealer')),
                upi_id TEXT NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_id INTEGER NOT NULL,
                receiver_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(sender_id) REFERENCES users(id),
                FOREIGN KEY(receiver_id) REFERENCES users(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dealer_id INTEGER NOT NULL,
                car_name TEXT NOT NULL,
                brand TEXT NOT NULL,
                year INTEGER NOT NULL,
                listing_type TEXT NOT NULL CHECK(listing_type IN ('Sell', 'Rent')),
                price REAL NOT NULL,
                photo_url TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(dealer_id) REFERENCES users(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS car_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_id INTEGER NOT NULL,
                customer_id INTEGER NOT NULL,
                dealer_id INTEGER NOT NULL,
                request_type TEXT NOT NULL CHECK(request_type IN ('Buy', 'Rent')),
                status TEXT NOT NULL DEFAULT 'Pending' CHECK(status IN ('Pending', 'Accepted', 'Rejected')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(car_id) REFERENCES cars(id),
                FOREIGN KEY(customer_id) REFERENCES users(id),
                FOREIGN KEY(dealer_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
        conn.close()
        print("SQLite database initialized.")
    except Exception as e:
        print(f"DB init error: {e}")

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

# Configure AI API
if AK and AK != "your_ai_api_key_here":
    genai.configure(api_key=AK)
    ai_model = genai.GenerativeModel('gemini-pro-latest')
    print("AI API configured successfully!")
else:
    print("AK not configured. Chatbot functionality will be limited.")
    print("To enable chatbot: Edit config.py and add your AI API key")
    ai_model = None

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

# -------------------- Connect (Auth + Chat) --------------------
@app.route('/connect')
def connect_home():
    if 'user_id' in session:
        return redirect(url_for('connect_chat'))
    return render_template('connect_home.html')

@app.route('/connect/signup', methods=['GET', 'POST'])
def connect_signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        account_type = request.form.get('account_type', '').strip()
        upi_id = request.form.get('upi_id', '').strip()
        password = request.form.get('password', '').strip()

        if not all([username, account_type, upi_id, password]):
            return render_template('connect_signup.html', error='All fields are required.')

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO users (username, account_type, upi_id, password) VALUES (?, ?, ?, ?)',
                (username, account_type, upi_id, password)
            )
            conn.commit()
            user_id = cur.lastrowid
            conn.close()
            session['user_id'] = user_id
            return redirect(url_for('connect_chat'))
        except sqlite3.IntegrityError:
            return render_template('connect_signup.html', error='Username already exists.',
                                   username=username, account_type=account_type, upi_id=upi_id)
        except Exception as e:
            return render_template('connect_signup.html', error='Error creating account.')

    return render_template('connect_signup.html')

@app.route('/connect/login', methods=['GET', 'POST'])
def connect_login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cur.fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('connect_chat'))
        return render_template('connect_login.html', error='Invalid credentials.', username=username)
    return render_template('connect_login.html')

@app.route('/connect/logout')
def connect_logout():
    session.clear()
    return redirect(url_for('connect_home'))

@app.route('/connect/delete_account', methods=['POST'])
def connect_delete_account():
    if 'user_id' not in session:
        return redirect(url_for('connect_login'))
    me_id = session['user_id']
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM messages WHERE sender_id = ? OR receiver_id = ?', (me_id, me_id))
        cur.execute('DELETE FROM users WHERE id = ?', (me_id,))
        conn.commit()
        conn.close()
    finally:
        session.clear()
    return redirect(url_for('connect_home'))

@app.route('/connect/chat')
def connect_chat():
    if 'user_id' not in session:
        return redirect(url_for('connect_login'))
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, account_type, upi_id FROM users ORDER BY username ASC')
    users = cur.fetchall()
    cur.execute('SELECT id, username, account_type, upi_id FROM users WHERE id = ?', (session['user_id'],))
    me = cur.fetchone()
    conn.close()
    return render_template('connect_chat.html', users=users, me=me)

@app.route('/connect/users')
def connect_users():
    if 'user_id' not in session:
        return jsonify([])
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, account_type, upi_id FROM users ORDER BY username ASC')
    users = [dict(u) for u in cur.fetchall()]
    conn.close()
    return jsonify(users)

@app.route('/connect/messages')
def connect_get_messages():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    other_id = request.args.get('user_id', type=int)
    if not other_id:
        return jsonify({'success': False, 'error': 'user_id is required'}), 400
    me_id = session['user_id']
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.id, m.sender_id, m.receiver_id, m.content, m.created_at,
               su.username AS sender_username, su.account_type AS sender_type, su.upi_id AS sender_upi,
               ru.username AS receiver_username, ru.account_type AS receiver_type, ru.upi_id AS receiver_upi
        FROM messages m
        JOIN users su ON su.id = m.sender_id
        JOIN users ru ON ru.id = m.receiver_id
        WHERE (m.sender_id = ? AND m.receiver_id = ?) OR (m.sender_id = ? AND m.receiver_id = ?)
        ORDER BY m.created_at ASC, m.id ASC
        """,
        (me_id, other_id, other_id, me_id)
    )
    rows = cur.fetchall()
    conn.close()
    return jsonify({'success': True, 'messages': [dict(r) for r in rows]})

@app.route('/connect/send', methods=['POST'])
def connect_send_message():
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    data = request.get_json(force=True)
    receiver_id = data.get('receiver_id')
    content = (data.get('content') or '').strip()
    if not receiver_id or not content:
        return jsonify({'success': False, 'error': 'receiver_id and content are required'}), 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('INSERT INTO messages (sender_id, receiver_id, content) VALUES (?, ?, ?)',
                (session['user_id'], int(receiver_id), content))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

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

        # Handle negative predictions
        if predicted_price < 0:
            return jsonify({
                'success': True,
                'predicted_price': None,
                'formatted_price': None,
                'message': 'This car does not have any value.'
            })
        
        # Format the price
        formatted_price = f"₹{predicted_price:,.0f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price,
            'message': 'Based on the provided information, this is the estimated market value of your vehicle.'
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

# -------------------- Rent/Buy Feature --------------------
@app.route('/rent_buy')
def rent_buy():
    """Rent/Buy main page"""
    if 'user_id' not in session:
        return redirect(url_for('connect_login'))
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, account_type, upi_id FROM users WHERE id = ?', (session['user_id'],))
    user = cur.fetchone()
    conn.close()
    
    if not user:
        session.clear()
        return redirect(url_for('connect_login'))
    
    # Generate a comprehensive year range (1950 to current year + 1)
    current_year = datetime.now().year
    year_range = list(range(1950, current_year + 2))  # +2 to include next year
    year_range.reverse()  # Most recent years first
    
    return render_template('rent_buy.html', user=dict(user), companies=companies, years=year_range)

@app.route('/rent_buy/listings')
def rent_buy_listings():
    """Get all car listings"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, u.username AS dealer_name, u.upi_id AS dealer_upi
        FROM cars c
        JOIN users u ON u.id = c.dealer_id
        ORDER BY c.created_at DESC
    """)
    listings = [dict(row) for row in cur.fetchall()]
    conn.close()
    return jsonify(listings)

@app.route('/rent_buy/my_listings')
def rent_buy_my_listings():
    """Get dealer's own listings"""
    if 'user_id' not in session:
        return jsonify([])
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, u.username AS dealer_name, u.upi_id AS dealer_upi
        FROM cars c
        JOIN users u ON u.id = c.dealer_id
        WHERE c.dealer_id = ?
        ORDER BY c.created_at DESC
    """, (session['user_id'],))
    listings = [dict(row) for row in cur.fetchall()]
    conn.close()
    return jsonify(listings)

@app.route('/rent_buy/create_listing', methods=['POST'])
def rent_buy_create_listing():
    """Create a car listing (Car Dealer only)"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT account_type FROM users WHERE id = ?', (session['user_id'],))
    user = cur.fetchone()
    
    if not user or user['account_type'] != 'Car Dealer':
        conn.close()
        return jsonify({'success': False, 'error': 'Only Car Dealers can create listings'}), 403
    
    data = request.get_json()
    car_name = data.get('car_name', '').strip()
    brand = data.get('brand', '').strip()
    year = int(data.get('year', 0))
    listing_type = data.get('listing_type', '').strip()
    price = float(data.get('price', 0))
    photo_url = data.get('photo_url', '').strip()
    description = data.get('description', '').strip()
    
    if not all([car_name, brand, year, listing_type, price]):
        conn.close()
        return jsonify({'success': False, 'error': 'All required fields must be filled'}), 400
    
    if listing_type not in ['Sell', 'Rent']:
        conn.close()
        return jsonify({'success': False, 'error': 'Invalid listing type'}), 400
    
    cur.execute("""
        INSERT INTO cars (dealer_id, car_name, brand, year, listing_type, price, photo_url, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (session['user_id'], car_name, brand, year, listing_type, price, photo_url, description))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/rent_buy/requests')
def rent_buy_requests():
    """Get requests for a dealer"""
    if 'user_id' not in session:
        return jsonify([])
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT cr.*, 
               c.car_name, c.brand, c.year, c.listing_type, c.price,
               cu.username AS customer_name, cu.upi_id AS customer_upi
        FROM car_requests cr
        JOIN cars c ON c.id = cr.car_id
        JOIN users cu ON cu.id = cr.customer_id
        WHERE cr.dealer_id = ?
        ORDER BY cr.created_at DESC
    """, (session['user_id'],))
    requests = [dict(row) for row in cur.fetchall()]
    conn.close()
    return jsonify(requests)

@app.route('/rent_buy/request', methods=['POST'])
def rent_buy_create_request():
    """Create a buy/rent request (Customer only)"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT account_type FROM users WHERE id = ?', (session['user_id'],))
    user = cur.fetchone()
    
    if not user or user['account_type'] != 'Customer':
        conn.close()
        return jsonify({'success': False, 'error': 'Only Customers can make requests'}), 403
    
    data = request.get_json()
    car_id = int(data.get('car_id', 0))
    request_type = data.get('request_type', '').strip()
    
    if not car_id or request_type not in ['Buy', 'Rent']:
        conn.close()
        return jsonify({'success': False, 'error': 'Invalid request'}), 400
    
    # Get car and dealer info
    cur.execute('SELECT dealer_id, listing_type FROM cars WHERE id = ?', (car_id,))
    car = cur.fetchone()
    if not car:
        conn.close()
        return jsonify({'success': False, 'error': 'Car not found'}), 404
    
    # Check if request already exists
    cur.execute("""
        SELECT id FROM car_requests 
        WHERE car_id = ? AND customer_id = ? AND status = 'Pending'
    """, (car_id, session['user_id']))
    existing = cur.fetchone()
    if existing:
        conn.close()
        return jsonify({'success': False, 'error': 'You already have a pending request for this car'}), 400
    
    cur.execute("""
        INSERT INTO car_requests (car_id, customer_id, dealer_id, request_type)
        VALUES (?, ?, ?, ?)
    """, (car_id, session['user_id'], car['dealer_id'], request_type))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/rent_buy/accept_request', methods=['POST'])
def rent_buy_accept_request():
    """Accept a request (Car Dealer only)"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT account_type FROM users WHERE id = ?', (session['user_id'],))
    user = cur.fetchone()
    
    if not user or user['account_type'] != 'Car Dealer':
        conn.close()
        return jsonify({'success': False, 'error': 'Only Car Dealers can accept requests'}), 403
    
    data = request.get_json()
    request_id = int(data.get('request_id', 0))
    action = data.get('action', '').strip()  # 'accept' or 'reject'
    
    if not request_id or action not in ['accept', 'reject']:
        conn.close()
        return jsonify({'success': False, 'error': 'Invalid request'}), 400
    
    # Verify the request belongs to this dealer
    cur.execute('SELECT dealer_id FROM car_requests WHERE id = ?', (request_id,))
    req = cur.fetchone()
    if not req or req['dealer_id'] != session['user_id']:
        conn.close()
        return jsonify({'success': False, 'error': 'Request not found or unauthorized'}), 404
    
    status = 'Accepted' if action == 'accept' else 'Rejected'
    cur.execute('UPDATE car_requests SET status = ? WHERE id = ?', (status, request_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/rent_buy/delete_listing', methods=['POST'])
def rent_buy_delete_listing():
    """Delete a car listing (Car Dealer only, own listings only)"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    user_id = session['user_id']
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT account_type FROM users WHERE id = ?', (user_id,))
    user = cur.fetchone()
    
    if not user or user['account_type'] != 'Car Dealer':
        conn.close()
        return jsonify({'success': False, 'error': 'Only Car Dealers can delete listings'}), 403
    
    data = request.get_json(silent=True) or {}
    try:
        listing_id = int(data.get('listing_id', 0))
    except (TypeError, ValueError):
        listing_id = 0
    
    if listing_id <= 0:
        conn.close()
        return jsonify({'success': False, 'error': 'Invalid listing ID'}), 400
    
    # Verify the listing belongs to this dealer
    cur.execute('SELECT id FROM cars WHERE id = ? AND dealer_id = ?', (listing_id, user_id))
    listing = cur.fetchone()
    if not listing:
        conn.close()
        return jsonify({'success': False, 'error': 'Listing not found or unauthorized'}), 404
    
    # Delete associated requests first (cascade delete)
    cur.execute('DELETE FROM car_requests WHERE car_id = ?', (listing_id,))
    # Delete the listing
    cur.execute('DELETE FROM cars WHERE id = ? AND dealer_id = ?', (listing_id, user_id))
    conn.commit()
    conn.close()
    
    return jsonify({'success': True})

@app.route('/rent_buy/accepted_requests')
def rent_buy_accepted_requests():
    """Get accepted requests history for a dealer"""
    if 'user_id' not in session:
        return jsonify([])
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT cr.*, 
               c.car_name, c.brand, c.year, c.listing_type, c.price,
               cu.username AS customer_name, cu.upi_id AS customer_upi
        FROM car_requests cr
        JOIN cars c ON c.id = cr.car_id
        JOIN users cu ON cu.id = cr.customer_id
        WHERE cr.dealer_id = ? AND cr.status = 'Accepted'
        ORDER BY cr.created_at DESC
    """, (session['user_id'],))
    requests = [dict(row) for row in cur.fetchall()]
    conn.close()
    return jsonify(requests)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', APP_PORT))
    init_db()
    app.run(debug=False, host='0.0.0.0', port=port)
