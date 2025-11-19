# Auto_Mind &mdash; Intelligent Automotive Assistant

Auto_Mind is a full-stack Flask application that blends machine learning, conversational guidance, and marketplace utilities for car buyers, sellers, and service providers. It ships with a polished dark UI, an AI-powered chatbot, and role-aware collaboration tools (Customers, Mechanics, Car Dealers) in a single deployable project.

---

## Table of Contents

1. [Key Features](#key-features)  
2. [Architecture at a Glance](#architecture-at-a-glance)  
3. [Screens & Workflows](#screens--workflows)  
4. [Getting Started](#getting-started)  
5. [Project Structure](#project-structure)  
6. [API Surface](#api-surface)  
7. [Development Tips](#development-tips)  
8. [License](#license)

---

## Key Features

- **Car Price Prediction**  
  Linear Regression model (`LinearRegressionModel.pkl`) estimates resale value using company, model, year, fuel type, and mileage. Dropdowns auto-populate from `Cleaned_Car.csv`.

- **Integrated EMI Planner**  
  On the Predict page, users can immediately simulate loan EMI, total payment, and interest distribution using the predicted price as a reference.

- **Conversational Assistant**  
  A modern chat interface relays automotive questions to an AI backend and renders responses with markdown support, typing indicators, and persistent history.

- **Connect Hub**  
  Multi-role authentication, one-to-one messaging, and account management keep customers, mechanics, and dealers in sync.

- **Rent/Buy Marketplace**  
  Dealers can create listings, manage inventory, review customer requests, and track accepted deals. Customers browse listings and raise buy/rent requests without seeing dealer-only actions.

- **Cohesive UI System**  
  Glassmorphism-inspired cards, smooth transitions, and responsive breakpoints are defined once in `static/styles.css` and reused across all templates.

---

## Architecture at a Glance

| Layer        | Highlights                                                                                  |
|-------------|----------------------------------------------------------------------------------------------|
| Backend     | Flask 2.x, SQLite, pandas/numpy, scikit-learn (price model), markdown rendering.             |
| Frontend    | Jinja2 templates, vanilla JS fetch/AJAX, Font Awesome icons, custom CSS variables & animations. |
| Intelligence| Pretrained Linear Regression model loaded at startup; datasets provide dropdown metadata.     |
| Sessions    | Flask session cookies protect authenticated routes (`FLASK_SECRET_KEY`).                     |
| Utilities   | `run.py` performs health checks before boot, `start.sh`/`start.bat` simplify launching.      |

---

## Screens & Workflows

1. **Home / Marketing Pages** – Static sections (`/`, `/services`, `/contact`, `/about`) introduce the platform.
2. **Predict** – Car price form + EMI calculator + animated result card.
3. **Chatbot** – Conversational UI with avatars, markdown support, and autosizing inputs.
4. **Connect** – Signup/login flow, user list, chat window, account deletion.
5. **Rent/Buy**  
   - _Dealer tabs_: My Listings, Create Listing, Pending Requests, Accepted History.  
   - _Customer view_: Marketplace grid with Buy/Rent request buttons (no delete actions visible).  
   - All destructive actions double-check role + ownership on the server.

---

## Getting Started

```bash
git clone <repo-url>
cd Auto_Mind
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

`run.py` validates that required assets (`LinearRegressionModel.pkl`, `Cleaned_Car.csv`, `templates/index.html`, `static/styles.css`) exist and then starts the Flask app on the configured port (default 5000).

### Environment Variables

| Variable           | Purpose                               |
|--------------------|---------------------------------------|
| `FLASK_SECRET_KEY` | Secures session cookies               |
| `PORT`             | Optional port override for the server |

> Additional provider-specific settings can be stored in `config.py` and imported by `app.py`.

---

## Project Structure

```
Auto_Mind/
├── app.py                    # Main Flask application & routes
├── run.py                    # Startup checks + dev server bootstrap
├── requirements.txt          # Python dependencies
├── config.py                 # Optional settings (imported if present)
├── start.sh / start.bat      # OS-friendly launchers
├── LinearRegressionModel.pkl # Serialized scikit-learn model
├── Cleaned_Car.csv           # Reference dataset for dropdowns
├── PROJECT_DOCUMENTATION.md  # In-depth technical guide
├── README.md                 # GitHub-friendly overview (this file)
├── templates/                # Jinja2 templates for every page
└── static/
    └── styles.css            # Global styling + animations
```

---

## API Surface

| Area           | Endpoint(s)                                   | Notes                                           |
|----------------|-----------------------------------------------|-------------------------------------------------|
| Public pages   | `GET /`, `/predict_page`, `/chatbot`, `/services`, `/contact`, `/about` | Rendered via Jinja2.                            |
| Prediction     | `GET /get_models/<company>`, `POST /predict`  | Dropdown hydration + price inference.           |
| Chatbot        | `POST /chat`                                  | Accepts JSON `{ "message": "..." }`.            |
| Auth           | `GET/POST /connect/signup`, `/connect/login`, `GET /connect/logout` | Session-backed authentication.                  |
| Messaging      | `GET /connect/chat`, `/connect/users`, `/connect/messages`, `POST /connect/send`, `POST /connect/delete_account` | Polling-based chat. |
| Rent/Buy       | `GET /rent_buy`, `/rent_buy/listings`, `/rent_buy/my_listings`, `/rent_buy/requests`, `/rent_buy/accepted_requests`, plus `POST /rent_buy/create_listing`, `/rent_buy/request`, `/rent_buy/accept_request`, `/rent_buy/delete_listing` | Role/ownership checks on every mutation. |

All mutation endpoints return JSON envelopes in the form `{ success: bool, ... }` with descriptive `error` messages when validation fails.

---

## Development Tips

- **Hot Reload**: `flask --app app --debug run` enables auto-reload and interactive debugger.
- **Database Reset**: Delete `app.db` to start fresh; `init_db()` will recreate tables on next launch.
- **Testing Workflows**: Use separate browsers/incognito windows to act as dealer vs. customer simultaneously, especially for the Rent/Buy feature.
- **Styling**: Reuse `var(--accent)`/`var(--card)` tokens when adding components to maintain UI consistency.

---

## License

This project is provided for educational and demonstration purposes. Feel free to adapt it for learning, prototypes, or internal tooling.
