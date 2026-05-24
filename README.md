# FlowCast

> Drop your bank file. See your financial future.

FlowCast turns messy transaction data into clear forecasts, spending insights, and money saving recommendations. No bank login, no spreadsheets, no setup.

## Live Demo

flowcast.samettemurcin.workers.dev

---

## Screenshots

<img width="1185" height="663" alt="Landing page " src="https://github.com/user-attachments/assets/4895cd5e-92d2-4313-9587-c9c6bdebda39" />

---

## Features

- **CSV Upload** : Works with any bank export (Chase, Bank of America, Wells Fargo, Citi)
- **Spending Dashboard** : Category breakdown, top merchants, day-of-week patterns
- **Time Periods** : Filter by This Month, 3M, 6M, or 1Y
- **Financial Forecast** : Prophet, ARIMA, and Linear Regression models with adjustable horizon
- **Anomaly Detection** : Flags unusual transactions automatically
- **Budget Tracking**: Overspend alerts and monthly summaries
- **Insights** : Plain-language recommendations based on your data

---

## Stack

| Layer | Technologies |
|---|---|
| Frontend | React, TypeScript, TanStack Router, Tailwind CSS |
| Charts | Recharts |
| Build | Vite |
| Hosting | Vercel |

---

## How to Run Locally

**Requirements:** Node.js 18+

```bash
# 1. Clone the repo
git clone https://github.com/samettemurcin/tryflowcast.git
cd tryflowcast

# 2. Install dependencies
npm install

# 3. Start dev server
npm run dev
```


---

## How to Use

1. Open the app at [tryflowcast.vercel.app](https://tryflowcast.vercel.app)
2. Click **Start Free** to go to the dashboard
3. Click **Upload Statement** and drop your bank CSV
4. Explore Dashboard, Spending, Forecast, and Insights pages

A sample CSV is available in the FlowCast backend repo for testing.

---

## Project Structure

```
src/
├── routes/
│   ├── index.tsx          # Landing page
│   ├── dashboard.tsx      # Main dashboard
│   ├── spending.tsx       # Spending analysis
│   ├── budget.tsx         # Budget tracking
│   ├── forecast.tsx       # Financial forecasting
│   └── insights.tsx       # AI insights
├── lib/
│   ├── forecast-models.ts # Prophet, ARIMA, Linear models
│   └── spending-aggregates.ts # CSV parsing and aggregation
└── components/
    └── ui/                # Shared UI components
```

---

## Related

- FlowCast Backend: Python analytics engine (Streamlit, ARIMA, Prophet)
