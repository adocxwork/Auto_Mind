import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import sqlite3
import google.generativeai as genai
from markdown import markdown
import re
from math import radians, sin, cos, sqrt, atan2

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
                password TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                address TEXT,
                phone TEXT
            )
            """
        )
        # Add location columns if they don't exist
        try:
            cur.execute("ALTER TABLE users ADD COLUMN latitude REAL")
            cur.execute("ALTER TABLE users ADD COLUMN longitude REAL")
            cur.execute("ALTER TABLE users ADD COLUMN address TEXT")
            cur.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        except sqlite3.OperationalError:
            pass  # Columns already exist
        
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
            CREATE TABLE IF NOT EXISTS cars_for_rent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_id INTEGER NOT NULL,
                company TEXT NOT NULL,
                model TEXT NOT NULL,
                year INTEGER NOT NULL,
                fuel_type TEXT NOT NULL,
                price_per_day REAL NOT NULL,
                location_latitude REAL,
                location_longitude REAL,
                location_address TEXT,
                description TEXT,
                image_url TEXT,
                available INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(owner_id) REFERENCES users(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rental_bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_id INTEGER NOT NULL,
                customer_id INTEGER NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                total_amount REAL NOT NULL,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending','confirmed','completed','cancelled')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(car_id) REFERENCES cars_for_rent(id),
                FOREIGN KEY(customer_id) REFERENCES users(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS custom_car_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                model TEXT NOT NULL,
                year INTEGER NOT NULL,
                fuel_type TEXT NOT NULL,
                price REAL NOT NULL,
                kms_driven INTEGER DEFAULT 0,
                image_url TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cars_for_sale (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dealer_id INTEGER NOT NULL,
                company TEXT NOT NULL,
                model TEXT NOT NULL,
                year INTEGER NOT NULL,
                fuel_type TEXT NOT NULL,
                price REAL NOT NULL,
                kms_driven INTEGER NOT NULL,
                location_latitude REAL,
                location_longitude REAL,
                location_address TEXT,
                description TEXT,
                image_url TEXT,
                available INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(dealer_id) REFERENCES users(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS car_bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_id INTEGER NOT NULL,
                customer_id INTEGER NOT NULL,
                booking_date DATE NOT NULL,
                booking_time TIME NOT NULL,
                contact_phone TEXT NOT NULL,
                contact_email TEXT,
                message TEXT,
                status TEXT DEFAULT 'pending' CHECK(status IN ('pending','confirmed','completed','cancelled')),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(car_id) REFERENCES cars_for_sale(id),
                FOREIGN KEY(customer_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()
        conn.close()
        print("SQLite database initialized.")
    except Exception as e:
        print(f"DB init error: {e}")

# Model removed - using dataset-based prediction only

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
    
    print("Dataset and company models loaded successfully!")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
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
        latitude = request.form.get('latitude', '').strip()
        longitude = request.form.get('longitude', '').strip()
        address = request.form.get('address', '').strip()
        phone = request.form.get('phone', '').strip()

        if not all([username, account_type, upi_id, password]):
            return render_template('connect_signup.html', error='All required fields are missing.')

        try:
            conn = get_db_connection()
            cur = conn.cursor()
            lat_val = float(latitude) if latitude else None
            lng_val = float(longitude) if longitude else None
            cur.execute(
                'INSERT INTO users (username, account_type, upi_id, password, latitude, longitude, address, phone) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (username, account_type, upi_id, password, lat_val, lng_val, address, phone)
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
            print(f"Signup error: {e}")
            return render_template('connect_signup.html', error='Error creating account.')

    account_type = request.args.get('type', '')
    return render_template('connect_signup.html', account_type=account_type)

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

def predict_price_from_dataset(company, car_model, year, kms_driven, fuel_type):
    """Predict car price using dataset statistics and intelligent calculations"""
    try:
        # Filter dataset for similar cars - try exact match first
        filtered = df[
            (df['company'] == company) & 
            (df['name'] == car_model) & 
            (df['fuel_type'] == fuel_type)
        ]
        
        if len(filtered) == 0:
            # Try with just company and fuel type
            filtered = df[
                (df['company'] == company) & 
                (df['fuel_type'] == fuel_type)
            ]
        
        if len(filtered) == 0:
            # Try with just company
            filtered = df[df['company'] == company]
        
        if len(filtered) == 0:
            return None
        
        # Calculate price based on similar cars with intelligent adjustments
        base_price = filtered['Price'].median()
        
        # Adjust for year (newer cars are more expensive, older depreciate)
        year_factor = 1.0
        median_year = filtered['year'].median()
        if year > median_year:
            year_diff = year - median_year
            year_factor = 1.0 + (year_diff * 0.025)  # 2.5% per year newer
        elif year < median_year:
            year_diff = median_year - year
            year_factor = 1.0 - (year_diff * 0.035)  # 3.5% depreciation per year older
        
        # Adjust for kilometers driven
        km_factor = 1.0
        avg_km = filtered['kms_driven'].median()
        if kms_driven > avg_km:
            km_diff = kms_driven - avg_km
            km_factor = 1.0 - (km_diff / 100000 * 0.12)  # 12% per 100k km
        elif kms_driven < avg_km:
            km_diff = avg_km - kms_driven
            km_factor = 1.0 + (km_diff / 100000 * 0.06)  # 6% bonus per 100k km less
        
        # Calculate final estimated price
        estimated_price = base_price * year_factor * max(km_factor, 0.25)  # Min 25% of base
        return max(estimated_price, 0)
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/predict', methods=['POST'])
def predict():
    """Handle car price prediction using dataset-based intelligent calculation"""
    try:
        # Get form data
        company = request.form.get('company', '').strip()
        car_model = request.form.get('car_model', '').strip()
        year_str = request.form.get('year', '').strip()
        kms_driven_str = request.form.get('kms_driven', '').strip()
        fuel_type = request.form.get('fuel_type', '').strip()
        
        # Validate inputs
        if not all([company, car_model, year_str, kms_driven_str, fuel_type]):
            return jsonify({'success': False, 'error': 'All fields are required'})
        
        try:
            year = int(year_str)
            kms_driven = int(kms_driven_str)
        except ValueError:
            return jsonify({'success': False, 'error': 'Year and kilometers must be valid numbers'})
        
        # Validate year range
        if year < 1900 or year > 2100:
            return jsonify({'success': False, 'error': 'Invalid year. Please enter a year between 1900 and 2100.'})
        
        # Validate kms_driven
        if kms_driven < 0:
            return jsonify({'success': False, 'error': 'Kilometers driven must be a positive number'})
        
        # Make prediction using dataset-based method
        predicted_price = predict_price_from_dataset(company, car_model, year, kms_driven, fuel_type)
        
        if predicted_price is None:
            return jsonify({
                'success': False, 
                'error': 'Could not find similar cars in our database. Please try different specifications.'
            })
        
        # Ensure reasonable range
        predicted_price = float(predicted_price)
        predicted_price = max(0, predicted_price)
        predicted_price = min(predicted_price, 50000000)  # Cap at 5 crores
        
        # Format the price
        formatted_price = f"₹{predicted_price:,.0f}"
        
        return jsonify({
            'success': True,
            'predicted_price': predicted_price,
            'formatted_price': formatted_price
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'})

@app.route('/recommendations')
def recommendations():
    """Car recommendation page with price ranges"""
    return render_template('recommendations.html', 
                         companies=companies, 
                         fuel_types=fuel_types, 
                         years=years)

@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    """Get car recommendations based on price range"""
    try:
        min_price = float(request.args.get('min_price', 0))
        max_price = float(request.args.get('max_price', 10000000))
        
        # Check what price column exists in the dataset
        price_col = None
        for col in ['selling_price', 'price', 'Price', 'Selling_Price']:
            if col in df.columns:
                price_col = col
                break
        
        if price_col:
            filtered_df = df[(df[price_col] >= min_price) & (df[price_col] <= max_price)]
        else:
            # If no price column, use prediction model to estimate prices
            # Sample from all cars and predict their prices
            sampled_df = df.sample(min(100, len(df)))
            predictions = []
            for _, row in sampled_df.iterrows():
                try:
                    prediction_data = pd.DataFrame([[
                        row['name'],
                        row['company'],
                        row['year'],
                        row['kms_driven'],
                        row['fuel_type']
                    ]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
                    pred_price = predict_price_from_dataset(
                        row['name'], row['company'], row['year'], 
                        row['kms_driven'], row['fuel_type']
                    )
                    if pred_price is None:
                        continue
                    pred_price = max(0, pred_price)
                    if min_price <= pred_price <= max_price:
                        # Convert Series to dict and add predicted price
                        row_dict = row.to_dict()
                        row_dict['predicted_price'] = pred_price
                        predictions.append(row_dict)
                except:
                    continue
            
            # Create DataFrame from predictions
            if predictions:
                filtered_df = pd.DataFrame(predictions)
                price_col = 'predicted_price'
            else:
                filtered_df = pd.DataFrame()
                price_col = None
        
        # Sample cars and add image URLs
        recommendations = []
        sample_size = min(20, len(filtered_df))
        if sample_size > 0:
            sampled_df = filtered_df.sample(n=sample_size) if len(filtered_df) > sample_size else filtered_df
            for _, row in sampled_df.iterrows():
                # Generate image URL - using Unsplash for car images
                car_query = f"{row['company'].lower().replace(' ', '+')}+{row['name'].lower().replace(' ', '+')}"
                image_url = f"https://source.unsplash.com/400x300/?car,{car_query}"
                
                # Fallback to placeholder if needed
                if not image_url or image_url.startswith('https://via'):
                    image_url = f"https://via.placeholder.com/400x300/667eea/ffffff?text={row['company']}+{row['name']}"
                
                price_value = float(row[price_col]) if price_col in row else 0
                
                recommendations.append({
                    'company': row['company'],
                    'model': row['name'],
                    'year': int(row['year']),
                    'fuel_type': row['fuel_type'],
                    'kms_driven': int(row['kms_driven']),
                    'price': price_value,
                    'image_url': image_url,
                    'is_custom': False
                })
        
        # Also include custom cars not in the dataset
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                SELECT company, model, year, fuel_type, price, kms_driven, image_url, description
                FROM custom_car_recommendations
                WHERE price >= ? AND price <= ?
                ORDER BY created_at DESC
            """, (min_price, max_price))
            custom_cars = cur.fetchall()
            conn.close()
            
            for car in custom_cars:
                # Generate image URL if not provided
                image_url = car['image_url']
                if not image_url:
                    car_query = f"{car['company'].lower().replace(' ', '+')}+{car['model'].lower().replace(' ', '+')}"
                    image_url = f"https://source.unsplash.com/400x300/?car,{car_query}"
                    if not image_url or image_url.startswith('https://via'):
                        image_url = f"https://via.placeholder.com/400x300/667eea/ffffff?text={car['company']}+{car['model']}"
                
                recommendations.append({
                    'company': car['company'],
                    'model': car['model'],
                    'year': int(car['year']),
                    'fuel_type': car['fuel_type'],
                    'kms_driven': int(car['kms_driven'] or 0),
                    'price': float(car['price']),
                    'image_url': image_url,
                    'is_custom': True,
                    'description': car['description'] or ''
                })
        except Exception as e:
            print(f"Error fetching custom cars: {e}")
        
        # Limit total recommendations to 30 (mix of dataset and custom cars)
        if len(recommendations) > 30:
            recommendations = recommendations[:30]
        
        return jsonify({'success': True, 'recommendations': recommendations})
    except Exception as e:
        print(f"Recommendation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recommendations/add_custom', methods=['POST'])
def api_add_custom_car():
    """Add a custom car recommendation not in the dataset"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['company', 'model', 'year', 'fuel_type', 'price']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'success': False, 'error': f'{field} is required'}), 400
        
        company = data.get('company').strip()
        model = data.get('model').strip()
        year = int(data.get('year'))
        fuel_type = data.get('fuel_type').strip()
        price = float(data.get('price'))
        kms_driven = int(data.get('kms_driven', 0))
        image_url = data.get('image_url', '').strip()
        description = data.get('description', '').strip()
        
        # Validate year
        if year < 1900 or year > 2100:
            return jsonify({'success': False, 'error': 'Invalid year'}), 400
        
        # Validate price
        if price < 0:
            return jsonify({'success': False, 'error': 'Price must be positive'}), 400
        
        # Generate image URL if not provided
        if not image_url:
            car_query = f"{company.lower().replace(' ', '+')}+{model.lower().replace(' ', '+')}"
            image_url = f"https://source.unsplash.com/400x300/?car,{car_query}"
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO custom_car_recommendations 
            (company, model, year, fuel_type, price, kms_driven, image_url, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (company, model, year, fuel_type, price, kms_driven, image_url, description))
        conn.commit()
        car_id = cur.lastrowid
        conn.close()
        
        return jsonify({'success': True, 'car_id': car_id, 'message': 'Custom car added successfully'})
    except ValueError as e:
        return jsonify({'success': False, 'error': f'Invalid data format: {str(e)}'}), 400
    except Exception as e:
        print(f"Add custom car error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/find_mechanics')
def find_mechanics():
    """Find nearest mechanics page"""
    if 'user_id' not in session:
        return redirect(url_for('connect_login'))
    return render_template('find_mechanics.html')

@app.route('/api/nearby_mechanics', methods=['GET'])
def api_nearby_mechanics():
    """Find nearby mechanics based on location"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        user_lat = float(request.args.get('latitude', 0))
        user_lng = float(request.args.get('longitude', 0))
        max_distance = float(request.args.get('max_distance', 50))  # km
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, username, account_type, latitude, longitude, address, phone, upi_id
            FROM users 
            WHERE account_type = 'Mechanic' 
            AND latitude IS NOT NULL 
            AND longitude IS NOT NULL
        """)
        mechanics = cur.fetchall()
        conn.close()
        
        nearby_mechanics = []
        for mechanic in mechanics:
            if mechanic['latitude'] and mechanic['longitude']:
                # Calculate distance using Haversine formula
                lat1, lon1 = radians(user_lat), radians(user_lng)
                lat2, lon2 = radians(mechanic['latitude']), radians(mechanic['longitude'])
                
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                distance = 6371 * c  # Earth radius in km
                
                if distance <= max_distance:
                    nearby_mechanics.append({
                        'id': mechanic['id'],
                        'username': mechanic['username'],
                        'address': mechanic['address'] or 'Address not provided',
                        'phone': mechanic['phone'] or 'Phone not provided',
                        'distance': round(distance, 2),
                        'latitude': mechanic['latitude'],
                        'longitude': mechanic['longitude']
                    })
        
        # Sort by distance
        nearby_mechanics.sort(key=lambda x: x['distance'])
        
        return jsonify({'success': True, 'mechanics': nearby_mechanics})
    except Exception as e:
        print(f"Mechanic finder error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/update_location', methods=['POST'])
def update_location():
    """Update user location"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        data = request.get_json()
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        address = data.get('address', '')
        phone = data.get('phone', '')
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE users 
            SET latitude = ?, longitude = ?, address = ?, phone = ?
            WHERE id = ?
        """, (latitude, longitude, address, phone, session['user_id']))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Location update error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/rental')
def rental_home():
    """Car rental service home page"""
    return render_template('rental.html', companies=companies, fuel_types=fuel_types)

@app.route('/api/rental/cars', methods=['GET'])
def api_rental_cars():
    """Get available cars for rent"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT c.*, u.username as owner_name
            FROM cars_for_rent c
            JOIN users u ON c.owner_id = u.id
            WHERE c.available = 1
            ORDER BY c.created_at DESC
        """)
        cars = cur.fetchall()
        conn.close()
        
        return jsonify({'success': True, 'cars': [dict(car) for car in cars]})
    except Exception as e:
        print(f"Rental cars error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rental/add_car', methods=['POST'])
def api_add_rental_car():
    """Add a car for rent"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        data = request.get_json()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO cars_for_rent 
            (owner_id, company, model, year, fuel_type, price_per_day, 
             location_latitude, location_longitude, location_address, description, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session['user_id'],
            data.get('company'),
            data.get('model'),
            int(data.get('year')),
            data.get('fuel_type'),
            float(data.get('price_per_day')),
            float(data.get('latitude', 0)),
            float(data.get('longitude', 0)),
            data.get('address', ''),
            data.get('description', ''),
            data.get('image_url', f"https://via.placeholder.com/400x300/667eea/ffffff?text={data.get('company')}")
        ))
        conn.commit()
        car_id = cur.lastrowid
        conn.close()
        
        return jsonify({'success': True, 'car_id': car_id})
    except Exception as e:
        print(f"Add rental car error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rental/book', methods=['POST'])
def api_rental_book():
    """Book a car for rent"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        data = request.get_json()
        from datetime import datetime
        
        start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')
        days = (end_date - start_date).days + 1
        
        # Get car price
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT price_per_day FROM cars_for_rent WHERE id = ?", (data.get('car_id'),))
        car = cur.fetchone()
        if not car:
            conn.close()
            return jsonify({'success': False, 'error': 'Car not found'})
        
        total_amount = float(car['price_per_day']) * days
        
        cur.execute("""
            INSERT INTO rental_bookings 
            (car_id, customer_id, start_date, end_date, total_amount, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
        """, (
            int(data.get('car_id')),
            session['user_id'],
            data.get('start_date'),
            data.get('end_date'),
            total_amount
        ))
        conn.commit()
        booking_id = cur.lastrowid
        conn.close()
        
        return jsonify({'success': True, 'booking_id': booking_id, 'total_amount': total_amount})
    except Exception as e:
        print(f"Rental booking error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/rental/my_bookings')
def api_my_bookings():
    """Get user's rental bookings"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT b.*, c.company, c.model, c.year, c.location_address, u.username as owner_name
            FROM rental_bookings b
            JOIN cars_for_rent c ON b.car_id = c.id
            JOIN users u ON c.owner_id = u.id
            WHERE b.customer_id = ?
            ORDER BY b.created_at DESC
        """, (session['user_id'],))
        bookings = cur.fetchall()
        conn.close()
        
        return jsonify({'success': True, 'bookings': [dict(b) for b in bookings]})
    except Exception as e:
        print(f"My bookings error: {e}")
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

# -------------------- Car Booking System (Dealership) --------------------
@app.route('/booking')
def booking_home():
    """Car booking service home page"""
    if 'user_id' not in session:
        return redirect(url_for('connect_login'))
    return render_template('booking.html', companies=companies, fuel_types=fuel_types)

@app.route('/api/booking/cars', methods=['GET'])
def api_booking_cars():
    """Get available cars for sale (booking)"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT c.*, u.username as dealer_name, u.phone as dealer_phone, u.address as dealer_address
            FROM cars_for_sale c
            JOIN users u ON c.dealer_id = u.id
            WHERE c.available = 1
            ORDER BY c.created_at DESC
        """)
        cars = cur.fetchall()
        conn.close()
        
        return jsonify({'success': True, 'cars': [dict(car) for car in cars]})
    except Exception as e:
        print(f"Booking cars error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/booking/add_car', methods=['POST'])
def api_add_booking_car():
    """Add a car for sale (dealers only)"""
    conn = None
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Check if user is a car dealer
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT account_type FROM users WHERE id = ?', (session['user_id'],))
        user = cur.fetchone()
        
        if not user or user['account_type'] != 'Car Dealer':
            conn.close()
            return jsonify({'success': False, 'error': 'Only car dealers can add cars for sale'}), 403
        
        data = request.get_json()
        
        # Use the same connection for the insert
        cur.execute("""
            INSERT INTO cars_for_sale 
            (dealer_id, company, model, year, fuel_type, price, kms_driven,
             location_latitude, location_longitude, location_address, description, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session['user_id'],
            data.get('company'),
            data.get('model'),
            int(data.get('year')),
            data.get('fuel_type'),
            float(data.get('price')),
            int(data.get('kms_driven', 0)),
            float(data.get('latitude', 0)) if data.get('latitude') else None,
            float(data.get('longitude', 0)) if data.get('longitude') else None,
            data.get('address', ''),
            data.get('description', ''),
            data.get('image_url', f"https://via.placeholder.com/400x300/667eea/ffffff?text={data.get('company')}+{data.get('model')}")
        ))
        conn.commit()
        car_id = cur.lastrowid
        conn.close()
        
        return jsonify({'success': True, 'car_id': car_id})
    except Exception as e:
        if conn:
            conn.close()
        print(f"Add booking car error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/booking/book_car', methods=['POST'])
def api_book_car():
    """Book a car for viewing/purchase"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        data = request.get_json()
        from datetime import datetime
        
        # Validate required fields
        if not all([data.get('car_id'), data.get('booking_date'), data.get('booking_time'), data.get('contact_phone')]):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Check if car exists and is available
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT available FROM cars_for_sale WHERE id = ?", (int(data.get('car_id')),))
        car = cur.fetchone()
        
        if not car:
            conn.close()
            return jsonify({'success': False, 'error': 'Car not found'})
        
        if not car['available']:
            conn.close()
            return jsonify({'success': False, 'error': 'Car is not available'})
        
        # Create booking
        cur.execute("""
            INSERT INTO car_bookings 
            (car_id, customer_id, booking_date, booking_time, contact_phone, contact_email, message, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            int(data.get('car_id')),
            session['user_id'],
            data.get('booking_date'),
            data.get('booking_time'),
            data.get('contact_phone'),
            data.get('contact_email', ''),
            data.get('message', '')
        ))
        conn.commit()
        booking_id = cur.lastrowid
        conn.close()
        
        return jsonify({'success': True, 'booking_id': booking_id, 'message': 'Booking request submitted successfully'})
    except Exception as e:
        print(f"Car booking error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/booking/my_bookings')
def api_my_car_bookings():
    """Get user's car bookings"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT b.*, c.company, c.model, c.year, c.price, c.location_address, 
                   u.username as dealer_name, u.phone as dealer_phone
            FROM car_bookings b
            JOIN cars_for_sale c ON b.car_id = c.id
            JOIN users u ON c.dealer_id = u.id
            WHERE b.customer_id = ?
            ORDER BY b.created_at DESC
        """, (session['user_id'],))
        bookings = cur.fetchall()
        conn.close()
        
        return jsonify({'success': True, 'bookings': [dict(b) for b in bookings]})
    except Exception as e:
        print(f"My car bookings error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/booking/dealer_bookings')
def api_dealer_bookings():
    """Get bookings for dealer's cars"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Check if user is a car dealer
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT account_type FROM users WHERE id = ?', (session['user_id'],))
        user = cur.fetchone()
        
        if not user or user['account_type'] != 'Car Dealer':
            conn.close()
            return jsonify({'success': False, 'error': 'Only car dealers can view dealer bookings'}), 403
        
        cur.execute("""
            SELECT b.*, c.company, c.model, c.year, c.price,
                   cu.username as customer_name, cu.phone as customer_phone
            FROM car_bookings b
            JOIN cars_for_sale c ON b.car_id = c.id
            JOIN users cu ON b.customer_id = cu.id
            WHERE c.dealer_id = ?
            ORDER BY b.created_at DESC
        """, (session['user_id'],))
        bookings = cur.fetchall()
        conn.close()
        
        return jsonify({'success': True, 'bookings': [dict(b) for b in bookings]})
    except Exception as e:
        print(f"Dealer bookings error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/booking/update_status', methods=['POST'])
def api_update_booking_status():
    """Update booking status (for dealers)"""
    try:
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        data = request.get_json()
        booking_id = data.get('booking_id')
        new_status = data.get('status')
        
        if new_status not in ['pending', 'confirmed', 'completed', 'cancelled']:
            return jsonify({'success': False, 'error': 'Invalid status'}), 400
        
        # Check if user is the dealer for this car
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT c.dealer_id 
            FROM car_bookings b
            JOIN cars_for_sale c ON b.car_id = c.id
            WHERE b.id = ?
        """, (booking_id,))
        result = cur.fetchone()
        
        if not result or result['dealer_id'] != session['user_id']:
            conn.close()
            return jsonify({'success': False, 'error': 'Unauthorized'}), 403
        
        cur.execute("""
            UPDATE car_bookings 
            SET status = ?
            WHERE id = ?
        """, (new_status, booking_id))
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Booking status updated'})
    except Exception as e:
        print(f"Update booking status error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', APP_PORT))
    init_db()
    app.run(debug=False, host='0.0.0.0', port=port)
