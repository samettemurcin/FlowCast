**FlowCast** 📈

Drop your bank file. See your financial future.

FlowCast turns messy transaction data into clear forecasts, spending insights, and money-saving recommendations no bank login required.

🌐 **Live Demo**:

🏠 Landing Page: https://myflowcast.lovable.app

🚀 App: https://tryflowcast.lovable.app

📸 **Screenshots**

<img width="1460" height="790" alt="Screenshot 2026-05-17 at 12 19 30 AM" src="https://github.com/user-attachments/assets/95bbddad-fb55-44cf-8dc2-3ca94807c260" />


✨ **Features**

1.Spending Breakdown by category, merchant, and day of week
2.6-24 Month Forecast Prophet + ARIMA with auto model selection
3.Budget Tracking overspend alerts and monthly summaries
4.Anomaly Detection flags transactions outside your normal pattern
5.Subscription Detector finds recurring charges you forgot about
6.Financial Health Score personalized recommendations based on your data


🛠 **Stack**
LayerTechnologiesFrontendReact, TypeScript, Tailwind CSSAnalyticsPython, Prophet, statsmodels, pandasHostingLovable, GitHub

🚀 **How to Run Locally**
Requirements: Python 3.9+
bash# 1. Clone the repo
git clone https://github.com/samettemurcin/FlowCast.git
cd FlowCast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
Then open your browser and go to http://localhost:8501
Try it with sample data
A sample_data.csv file is included in the repo — drop it into the app to see a full demo without needing your own bank file.

📂 **Project Structure**
FlowCast/
├── app.py              # Main application entry point
├── data_processor.py   # CSV parsing, cleaning, and categorization
├── forecaster.py       # ARIMA + Prophet forecasting logic
├── charts.py           # Visualization components
├── requirements.txt    # Python dependencies
└── sample_data.csv     # Sample bank export for testing

🎯 **Who It's For**
Built for US users managing personal finances or running a small business. Works with any CSV or Excel export from Chase, Bank of America, Wells Fargo, Citi, or any US bank.

👥 Contributors
Built as a collaborative project. Contributions welcome open an issue or submit a PR.
