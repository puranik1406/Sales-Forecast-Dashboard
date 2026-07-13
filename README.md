# 📈 End-to-End Sales Forecasting & Demand Intelligence System

An interactive Streamlit dashboard for sales analysis, forecasting, anomaly detection, and product demand segmentation using machine learning and time series forecasting techniques.

## 🚀 Features

### 📊 Sales Overview
- Year-wise total sales visualization
- Monthly sales trend analysis
- Interactive filtering by Region and Category
- Key business metrics including:
  - Total Sales
  - Average Order Value
  - Number of Orders

### 🔮 Forecast Explorer
- Forecast sales using Facebook Prophet
- Forecast by:
  - Category
  - Region
- Adjustable forecast horizon (1–3 months)
- Displays:
  - Forecasted sales
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

### ⚠️ Anomaly Report
- Detects anomalies in weekly sales using Isolation Forest
- Interactive visualization of anomalies
- Detailed anomaly table with week numbers

### 📦 Product Demand Segments
- Customer demand segmentation using K-Means Clustering
- Feature engineering using:
  - Total Sales
  - Growth Rate
  - Sales Volatility
  - Average Sales
- PCA-based visualization of clusters
- Cluster assignment for each product sub-category

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Prophet
- Scikit-learn
- Plotly

---

## 📂 Dataset

The project uses the **Superstore Sales Dataset**, containing historical sales records including:

- Orders
- Products
- Categories
- Regions
- Sales
- Profit
- Shipping information

---

## 📊 Machine Learning Models

| Task | Model |
|------|-------|
| Sales Forecasting | Prophet |
| Anomaly Detection | Isolation Forest |
| Demand Segmentation | K-Means Clustering |
| Visualization | PCA |

---

## 📁 Project Structure

```
.
├── app.py
├── train.csv
├── requirements.txt
└── README.md
```

---

## ▶️ Running the Project

### Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 📷 Dashboard Pages

- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

---

## 👤 Author

**Ishita Puranik**

Developed as part of an End-to-End Sales Forecasting & Demand Intelligence project using Streamlit and Machine Learning.
