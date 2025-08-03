import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib

# Load Models and Data

@st.cache_data
def load_models_and_data():
    scaler = joblib.load('models/rfm_scaler.pkl')
    kmeans = joblib.load('models/rfm_kmeans_model.pkl')
    similarity_matrix = joblib.load('similarity_matrix.pkl')
    df = pd.read_csv('RFM_Features.csv')
    # Transactional data for product recommendation
    products_df = pd.read_csv('cleaned_Shopper_Spectrum_data.csv')
    return scaler, kmeans, similarity_matrix, df, products_df


scaler, kmeans, similarity_matrix, df, products_df  = load_models_and_data()

# Cluster Label Mapping (sorted by Monetary)

@st.cache_resource
def get_cluster_labels():
    rfm_df = pd.read_csv("RFM_Features.csv")
    cluster_order = (
        rfm_df.groupby("Cluster")["Monetary"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    return {
        cluster_order[0]: 'High-Value',
        cluster_order[1]: 'Regular',
        cluster_order[2]: 'Occasional',
        cluster_order[3]: 'At-Risk'
    }

cluster_labels = get_cluster_labels()

# Prediction Function

def predict_rfm_segment(recency, frequency, monetary):
    input_data = scaler.transform([[recency, frequency, monetary]])
    cluster = kmeans.predict(input_data)[0]
    return cluster_labels[cluster]

# Recommendation Function

def recommend_products(product_name, num_recommendations=5):
    # Check if the product exists in similarity matrix
    if product_name not in similarity_matrix.columns:
        return []

    # Get the similarity scores for that product
    similar_scores = similarity_matrix[product_name].sort_values(ascending=False)

    # Exclude the product itself and return top N recommendations
    recommended_products = list(similar_scores.iloc[1:num_recommendations+1].index)
    return recommended_products


# Streamlit UI

st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("Shopper Spectrum: Customer Segmentation + Product Recommender")

st.markdown("This app helps you **segment customers using RFM analysis** and also **recommend similar products** based on past purchase patterns.")

# Sidebar
st.sidebar.title("Select Feature")

feature = st.sidebar.radio("Choose Feature:", ["Customer Segmentation", "Product Recommendation"])

# Feature 1: Customer Segmentation

if feature == "Customer Segmentation":
    st.header("RFM-Based Customer Segment Prediction")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, step=1)
    frequency = st.number_input("Frequency (total purchases)", min_value=1, max_value=1000, step=1)
    monetary = st.number_input("Monetary (total spent in $)", min_value=1.0, step=1.0)

    if st.button("Predict Segment"):
        segment = predict_rfm_segment(recency, frequency, monetary)
        st.success(f"The customer belongs to the **{segment}** segment.")

# Feature 2: Product Recommendation

elif feature == "Product Recommendation":
    st.header("Product-Based Recommendation System")

    product_list = products_df["Description"].dropna().unique()
    selected_product = st.selectbox("Select a Product", sorted(product_list))

    if st.button("Get Recommendations"):
        recommendations = recommend_products(selected_product)
        if recommendations:
            st.subheader("Recommended Products:")
            for idx, product in enumerate(recommendations, start=1):
                st.markdown(f"{idx}. {product}")
        else:
            st.warning("No recommendations found. Try another product.")
