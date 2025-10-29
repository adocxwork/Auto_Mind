#!/usr/bin/env python3
"""
Startup script for Auto_Mind Flask application
"""

import os
import sys
from app import app

# Import configuration
try:
    from config import APP_HOST, APP_PORT
except ImportError:
    APP_HOST = '0.0.0.0'
    APP_PORT = 5000

def main():
    """Start the Flask application"""
    print("Starting Auto_Mind Flask Application...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'LinearRegressionModel.pkl',
        'Cleaned_Car.csv',
        'templates/index.html',
        'static/styles.css'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are in the project directory.")
        return False
    
    # Check for AI API key
    try:
        from config import AK
        if not AK or AK == "your_ai_api_key_here":
            print("⚠️  Warning: AK not configured. Chatbot functionality will be limited.")
            print("   To enable full chatbot functionality, edit config.py and add your API key.")
            print("   Get your API key from: https://makersuite.google.com/app/apikey")
            print()
    except ImportError:
        print("⚠️  Warning: config.py not found. Using default settings.")
        print()
    
    print("✅ All required files found!")
    print("🚀 Starting Flask application...")
    print(f"📱 Application will be available at: http://localhost:{APP_PORT}")
    print("🛑 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        app.run(debug=True, host=APP_HOST, port=APP_PORT)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return False

if __name__ == "__main__":
    main()
