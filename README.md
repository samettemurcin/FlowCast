# FlowCast 📈

> Drop your bank file. See your financial future.

FlowCast turns messy transaction data into clear forecasts, spending insights, and money-saving recommendations no bank login required.

---

## 🌐 Live Demo

- 🏠 Landing Page: [myflowcast.lovable.app](https://myflowcast.lovable.app)
- 🚀 App: [tryflowcast.lovable.app](https://tryflowcast.lovable.app)

---

## 📸 Screenshots

<img width="1299" height="784" alt="Screenshot 2026-05-17 at 12 33 23 AM" src="https://github.com/user-attachments/assets/b795abdd-48bc-468c-8bbd-85a1eb3f301b" />


---

## ✨ Features

- **Spending Breakdown** : by category, merchant, and day of week
- **6–24 Month Forecast** : Prophet + ARIMA with auto model selection
- **Budget Tracking** : overspend alerts and monthly summaries
- **Anomaly Detection** : flags transactions outside your normal pattern
- **Subscription Detector** : finds recurring charges you forgot about
- **Financial Health Score** : personalized recommendations based on your data

---

## 🛠 Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, Tailwind CSS |
| Analytics | Python, Prophet, statsmodels, pandas |
| Hosting | Lovable, GitHub |

---

## 🚀 How to Run Locally

**Requirements:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/samettemurcin/FlowCast.git
cd FlowCast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open your browser and go to `http://localhost:8501`

> A `sample_data.csv` file is included — drop it into the app to see a full demo without needing your own bank file.

---

## 📂 Project Structure

```
FlowCast/
├── app.py              # Main application entry point
├── data_processor.py   # CSV parsing, cleaning, and categorization
├── forecaster.py       # ARIMA + Prophet forecasting logic
├── charts.py           # Visualization components
├── requirements.txt    # Python dependencies
└── sample_data.csv     # Sample bank export for testing
```

---

## 🎯 Who It's For

Built for US users managing personal finances or running a small business. Works with any CSV or Excel export from Chase, Bank of America, Wells Fargo, Citi, or any US bank.

---

## 👥 Contributors

Built as a collaborative project. Contributions welcome open an issue or submit a PR.

Built as a collaborative project. Contributions welcome open an issue or submit a PR.
