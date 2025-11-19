# Auto_Mind – Detailed Technical Documentation

_Last updated: November 2025_

---

## 1. Solution Overview

Auto_Mind is a full-stack automotive platform that merges three problem spaces into a single Flask application:

1. **Intelligence** – AI-assisted car price prediction and conversational guidance.  
2. **Marketplace** – Dealer-driven rent/buy listings with lead capture, EMI planning, and request workflows.  
3. **Community** – Secure user onboarding plus real-time messaging bridging customers, mechanics, and dealers.

Every module ships in one deployable unit (`app.py`) backed by SQLite, a serialized ML model, and static assets under `templates/` and `static/`.

---

## 2. High-Level Architecture

| Layer            | Components / Files                              | Notes                                                                                           |
|------------------|-------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Presentation     | `templates/*.html`, `static/styles.css`         | Jinja2 templating, responsive layouts, Font Awesome icons, animated interactions, EMI widget.    |
| Application      | `app.py`, `run.py`                              | Flask routes for public pages, prediction API, chatbot, Connect module, Rent/Buy workflows.      |
| Intelligence     | `LinearRegressionModel.pkl`, `Cleaned_Car.csv`  | scikit-learn linear regression model plus curated dataset powering dropdowns and predictions.    |
| Persistence      | `app.db` (SQLite)                               | Tables: `users`, `messages`, `cars`, `car_requests`. Managed through helper `get_db_connection`. |
| Automation       | `start.sh`, `start.bat`, `requirements.txt`     | One-command startup (env checks in `run.py`), dependency pinning, multi-OS launch scripts.       |

All requests travel through Flask; there is no client-side routing. Sessions (secured via `FLASK_SECRET_KEY`) gate authenticated experiences.

---

## 3. Module Deep Dive

### 3.1 Car Price Prediction (`/predict_page`, `/predict`, `/get_models/<company>`)

* **Dataset bootstrap** – On start, `Cleaned_Car.csv` is parsed via pandas, producing:
  * `companies`, `fuel_types`, `years` arrays for dropdowns.
  * `company_models` map to populate the “Model” list dynamically through AJAX.
* **Model lifecycle** – `LinearRegressionModel.pkl` is deserialized once. If loading fails, `/predict` responds with `success: False`.
* **User experience** – `templates/predict.html` hosts the form, animated inputs, an EMI calculator, and result cards.
* **Request flow**  
  1. Client submits `FormData` to `POST /predict`.  
  2. Backend validates payload, builds a single-row DataFrame, and calls `model.predict(...)`.  
  3. Results are formatted (`₹`, thousand separators) and pushed back as JSON for the UI to animate into place.
* **Edge handling** – Negative predictions return a “no market value” message rather than exposing raw numbers.

### 3.2 EMI Calculator (Predict Page)

Pure front-end feature sharing the styling system:

* Inputs: loan amount, annual rate, tenure (years).
* Uses `Intl.NumberFormat('en-IN')` for currency consistency.
* Shows monthly EMI, total payment, total interest, tenure (months), and a dual-progress bar that visualizes principal vs. interest proportions.
* Tied into the prediction narrative (“use the predicted price as loan amount”).

### 3.3 Chatbot (`/chatbot`, `POST /chat`)

* UI (`templates/chatbot.html`) mimics modern chat apps with avatars, typing indicator, markdown rendering, and scrollable history.
* Backend constructs a prompt contextualizing Auto_Mind’s domain, forwards the user question to the AI provider, converts markdown to HTML via `markdown` package, then returns JSON.
* Graceful failure – if the AI model is unavailable (e.g., missing credentials), the endpoint responds with a helpful message instead of crashing the page.

### 3.4 Connect Module (Auth + Messaging)

**Routes**: `/connect`, `/connect/signup`, `/connect/login`, `/connect/chat`, `/connect/users`, `/connect/messages`, `/connect/send`, `/connect/delete_account`.

* **Roles** – Customer, Mechanic, Car Dealer. Chosen at signup and stored in the `users` table together with `upi_id`.
* **Session handling** – `session['user_id']` drives authorization. Logout and account deletion both clear the session to avoid stale auth.
* **Messaging**  
  * Front-end polls `/connect/messages?user_id=<peer>` every ~2 seconds (AJAX) to keep the conversation fresh.
  * Messages persist with timestamps; both sender and receiver metadata are returned for UI labeling.
* **Account deletion** – Removes user and all associated messages before ending the session.

### 3.5 Rent/Buy Marketplace

**Routes**: `/rent_buy`, `/rent_buy/listings`, `/rent_buy/my_listings`, `/rent_buy/create_listing`, `/rent_buy/request`, `/rent_buy/requests`, `/rent_buy/accept_request`, `/rent_buy/delete_listing`, `/rent_buy/accepted_requests`.

Key behaviors:

* **Dealer dashboard** – Tabbed SPA-like interface inside `templates/rent_buy.html`:
  * _My Listings_: grid of the dealer’s cars with delete controls.
  * _Create Listing_: form for `car_name`, `brand`, `year`, `listing_type` (Sell/Rent), `price`, optional photo/description.
  * _Requests_: view inbound buy/rent requests, with Accept/Reject actions.
  * _Accepted History_: audit of past accepted deals.
* **Customer view** – Read-only grid of all listings plus “Request to Buy/Rent” buttons. No delete controls are rendered (enforced both in UI and backend).
* **Server-side guards** – Creation, deletion, and request acceptance all verify the user’s role and ownership. Deleting a listing cascades to `car_requests` before removing the car to preserve referential integrity.
* **Request flow** – Customers choose Buy or Rent; duplicate pending requests are blocked; dealers respond and status updates reflect in UI.

---

## 4. Data Model

```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    account_type TEXT NOT NULL CHECK(account_type IN ('Customer','Mechanic','Car Dealer')),
    upi_id TEXT NOT NULL,
    password TEXT NOT NULL
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender_id INTEGER NOT NULL,
    receiver_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(sender_id) REFERENCES users(id),
    FOREIGN KEY(receiver_id) REFERENCES users(id)
);

CREATE TABLE cars (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dealer_id INTEGER NOT NULL,
    car_name TEXT NOT NULL,
    brand TEXT NOT NULL,
    year INTEGER NOT NULL,
    listing_type TEXT NOT NULL CHECK(listing_type IN ('Sell','Rent')),
    price REAL NOT NULL,
    photo_url TEXT,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(dealer_id) REFERENCES users(id)
);

CREATE TABLE car_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    car_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    dealer_id INTEGER NOT NULL,
    request_type TEXT NOT NULL CHECK(request_type IN ('Buy','Rent')),
    status TEXT NOT NULL DEFAULT 'Pending' CHECK(status IN ('Pending','Accepted','Rejected')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(car_id) REFERENCES cars(id),
    FOREIGN KEY(customer_id) REFERENCES users(id),
    FOREIGN KEY(dealer_id) REFERENCES users(id)
);
```

Indexes rely on SQLite’s implicit PK indexes. Given the local deployment scope, additional tuning is unnecessary.

---

## 5. Request & Response Contracts (Representative)

| Endpoint | Method | Payload / Params                            | Response (200)                                                |
|----------|--------|---------------------------------------------|----------------------------------------------------------------|
| `/predict` | POST | `company`, `car_model`, `year`, `kms_driven`, `fuel_type` (form-data) | `{ success, formatted_price, message }` or `{ success:false, error }` |
| `/chat` | POST | `{ "message": "..." }` JSON | `{ success:true, response:"<p>…</p>" }` |
| `/rent_buy/create_listing` | POST | JSON body with car fields       | `{ success:true }` / `{ success:false, error }` |
| `/rent_buy/request` | POST | `{ car_id, request_type }`         | `{ success:true }` / validation errors                         |
| `/rent_buy/delete_listing` | POST | `{ listing_id }`                | `{ success:true }` (only for owning dealers)                   |
| `/connect/messages` | GET | `?user_id=<peer>` query              | `{ success:true, messages:[...] }` or 4xx on auth failure      |

All JSON endpoints rely on `fetch` with `Content-Type: application/json` (except form submissions which use `FormData`).

---

## 6. Front-End Experience

* **Design system** – Single `styles.css` defines CSS variables, gradients, glassmorphism cards, responsive grids, and animations (e.g., `fadeInUp`, `slideInUp`, floating background elements).
* **Navigation** – Sticky translucent navbar with mobile slide-out controlled via `mobileToggle`.
* **Predict page** – Combines ML form + EMI + result card for a cohesive “insight” section.
* **Rent/Buy** – Reuses prediction card aesthetics for marketplace cards, keeping consistent theming.
* **Connect chat** – Split-pane layout (sidebar list + chat area) optimized for desktop; collapses gracefully on <900px.
* **Accessibility** – Buttons include icons + text; forms use labels; color contrasts are tuned for dark themes.

---

## 7. Initialization & Configuration

1. **Dependencies** – `pip install -r requirements.txt`
2. **Database** – `init_db()` runs automatically on start; `app.db` will be created with all tables/checks.
3. **Model & dataset** – Ensure `LinearRegressionModel.pkl` and `Cleaned_Car.csv` remain in project root.
4. **Secrets & Ports** – Environment variables:
   * `FLASK_SECRET_KEY` (session encryption)
   * `PORT` (optional override for Flask host port)
   * Any additional keys needed by optional integrations (see `config.py`).
5. **Launch** – Use `python run.py` for a safety-checked startup (verifies required files and surfaces helpful warnings).

---

## 8. Operational Considerations

* **Security** – Role checks and ownership validation are enforced server-side on all state-changing routes. Even if the UI is manipulated, non-dealers cannot delete cars, and customers cannot accept requests.
* **Resilience** – Failure to load ML model or AI client is logged and gracefully surfaced to the user without crashing the app.
* **Scalability** – Current design targets single-node deployments. For production, you would:
  * Swap SQLite for PostgreSQL/MySQL.
  * Move model serving into a dedicated service (or use a standardized ML inference endpoint).
  * Introduce WebSocket-based messaging instead of polling.
* **Extensibility** – All feature blocks live in distinct template sections and Flask endpoints, making it straightforward to bolt on analytics, payments, or inventory importers.

---

## 9. Developer Tips

* **Hot reload** – Run `flask --app app --debug run` during development for autoreload.
* **Data resets** – Delete `app.db` to recreate a blank database; `init_db()` is idempotent.
* **Testing interactions** – The Rent/Buy feature depends on multiple roles. Use multiple browser sessions or incognito windows to simulate customer vs. dealer behavior simultaneously.
* **Styling** – Keep new components within the existing CSS token system (`--primary`, `--accent`, etc.) to preserve visual consistency.

---

## 10. Summary

Auto_Mind is more than a prediction demo—it is a cohesive automotive assistant that:

* Predicts car prices with a trained ML model.  
* Provides actionable financing context via an EMI calculator.  
* Enables conversational support through an AI chatbot.  
* Connects stakeholders via role-aware messaging.  
* Powers a rent/buy marketplace with request workflows, audit history, and safeguards.

The project illustrates how traditional Flask apps can blend data science outputs, modern UI polish, and collaborative utilities inside one maintainable codebase.
