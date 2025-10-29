# Auto_Mind - AI Car Price Prediction Platform

A Flask-based web application that uses machine learning to predict car prices and includes an AI chatbot powered by Google's AI API.

## Features

- **AI-Powered Car Price Prediction**: Uses a trained Linear Regression model to predict car prices based on various features
- **Interactive Chatbot**: AI assistant powered by Google's AI API for car-related queries
- **Modern UI**: Responsive design with dark theme and smooth animations
- **Dynamic Form**: Company-based model selection with real-time updates

## Project Structure

```
Auto_Mind/
├── app.py                    # Main Flask application
├── config.py                 # Configuration file (API keys, settings)
├── run.py                    # Enhanced startup script with checks
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
├── start.bat                 # Windows startup script
├── start.sh                  # Linux/Mac startup script
├── LinearRegressionModel.pkl # Trained ML model
├── Cleaned_Car.csv           # Dataset for dropdown options
├── templates/                 # HTML templates
│   ├── index.html           # Home page
│   ├── predict.html         # Price prediction form
│   ├── chatbot.html         # AI assistant
│   ├── services.html        # Services page
│   ├── contact.html         # Contact page
│   └── about.html           # About page
└── static/                  # Static files
    └── styles.css           # Main stylesheet
```

## Model Information

The application uses a Linear Regression model trained on car data with the following features:
- **Company**: Car manufacturer
- **Model**: Car model name
- **Year**: Manufacturing year
- **Kilometers Driven**: Mileage
- **Fuel Type**: Petrol, Diesel, or LPG

The model uses One-Hot Encoding for categorical features and has been optimized for accuracy.

## Application Pages

- **Home** (`/`) - Landing page with overview
- **Predict** (`/predict_page`) - Car price prediction form
- **Chat** (`/chatbot`) - AI assistant for car-related queries
- **Services** (`/services`) - Available services
- **Contact** (`/contact`) - Contact information
- **About** (`/about`) - About the application

## API Endpoints

- `GET /` - Home page
- `GET /predict_page` - Prediction form
- `GET /chatbot` - AI chatbot interface
- `POST /predict` - Car price prediction
- `POST /chat` - Chatbot messages
- `GET /get_models/<company>` - Get models for a company

## Troubleshooting

1. **Model not loading**: Ensure `LinearRegressionModel.pkl` is in the project root
2. **Dropdowns empty**: Verify `Cleaned_Car.csv` is in the project root
3. **Port already in use**: Change the port in `config.py` or kill the process using port 5000

## Development

To run in development mode:
```bash
export FLASK_ENV=development
python app.py
```

## License

This project is for educational purposes.
