# Shopper Spectrum: Customer Segmentation & Product Recommendation

This project performs intelligent **customer segmentation** using **RFM analysis** and **KMeans clustering**, followed by a **product recommendation engine** based on customer behavior. The entire pipeline is visualized and deployed as an interactive **Streamlit web app**.

---

## Demo

[Click here to launch the app]([https://your-app-name.streamlit.app](https://ecommerce-customer-segmentation-recommender.streamlit.app/)) (hosted on Streamlit Cloud)

---

## Project Objectives

- Segment customers based on Recency, Frequency, and Monetary value
- Discover actionable customer clusters via KMeans
- Recommend products tailored to selected customer behavior
- Visualize insights and clusters in an interactive app

---

## Dataset

Used a **retail transactions dataset** containing:

- `InvoiceNo`: Transaction ID
- `CustomerID`: Unique customer
- `InvoiceDate`: Date of purchase
- `StockCode` & `Description`: Products purchased
- `Quantity` & `UnitPrice`: Order values
- `Country`: Geographic data

---

## 1. Data Cleaning

Cleaned raw dataset using:

- Removing null `CustomerID`s
- Removing canceled orders (`InvoiceNo` starting with "C")
- Removing negative/zero `Quantity` or `UnitPrice`
- Feature engineering:
  - `TotalAmount = Quantity × UnitPrice`
  - `Date` features from `InvoiceDate`

---

## 2. EDA & RFM Analysis

Performed **Exploratory Data Analysis**:

- Top selling products & countries
- Distribution of revenue and customer contribution

### RFM (Recency, Frequency, Monetary) Feature Engineering:

- **Recency**: Days since last purchase
- **Frequency**: Number of orders per customer
- **Monetary**: Total spent by customer

All RFM features were:

- Scaled using **MinMaxScaler**
- Stored for clustering and downstream usage

---

## 3. Customer Segmentation using KMeans

### Preprocessing

- Used the scaled RFM matrix
- Used the **Elbow Method** to select optimal clusters (k)
- Chose `k=4` for clear differentiation

### KMeans Clustering

- Applied `KMeans` on scaled RFM data
- Saved model using `joblib`
- Analyzed each cluster's traits:
  - Loyal high-spenders
  - New or occasional buyers
  - Low spenders
  - At-risk customers

### Visualizations

- Cluster plots (2D)
- RFM distributions per cluster
- Cluster-based behavior analysis

---

## 4. Product Recommendation System

- Built a **customer-product matrix**
- For a selected product, computed **cosine similarity** with others
- Recommended top 5 similar products based on co-occurrence across invoices
- Built for **practical retail targeting**

---

## 5. Streamlit App

### Features:

- Interactive customer cluster visualization
- Product recommendation engine
- Real-time exploration of user segments
- Simple sidebar-based UI for navigation

### Deployment:

- App hosted via **[Streamlit Cloud]([https://streamlit.io/cloud](https://ecommerce-customer-segmentation-recommender.streamlit.app/))**  
- No setup required — runs directly in the browser

---

---

## 🛠Tech Stack

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Joblib**

---
