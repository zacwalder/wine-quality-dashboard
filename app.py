
"""
🍷 Wine Quality Prediction Dashboard
MSc Business Analytics — AUEB
Course: Machine Learning and Content Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import plotly.express as px
import plotly.graph_objects as go
import os
import traceback

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM STYLING
# ================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #722F37;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8F0E3;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background-color: #722F37;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# LOAD MODELS AND DATA (CACHED)
# ================================
@st.cache_resource
def load_artifacts():
    """Load all trained models and preprocessing artifacts."""
   
    regressor = joblib.load('models/best_xgb_regressor.pkl')
    classifier = joblib.load('models/best_xgb_classifier.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')

    return regressor, classifier, scaler, label_encoder, feature_names

@st.cache_data
def load_wine_data():
    """Load the wine dataset."""
    return pd.read_csv('data/wine_data.csv')


try:
    regressor, classifier, scaler, label_encoder, feature_names = load_artifacts()
    wine_df = load_wine_data()
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading files")
    st.error(f"**Error type:** {type(e).__name__}")
    st.error(f"**Error message:** {str(e)}")
    
    # Show what files actually exist in the repo
    st.markdown("### 📁 Files found in this deployment:")
    for folder in ['.', 'models', 'data']:
        if os.path.exists(folder):
            files = os.listdir(folder)
            st.markdown(f"**{folder}/**")
            for f in files:
                st.code(f"{folder}/{f}")
        else:
            st.warning(f"⚠️ Folder `{folder}/` does not exist")
    
    st.markdown("### 🐛 Full traceback:")
    st.code(traceback.format_exc())
    
    st.stop()  # ← This prevents the rest of the app from running


# ================================
# HEADER
# ================================
st.markdown('<h1 class="main-header">🍷 Wine Quality Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict wine quality from physicochemical properties using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ================================
# SIDEBAR — NAVIGATION
# ================================
st.sidebar.image("https://img.icons8.com/emoji/96/wine-glass-emoji.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "🔮 Predictor", "📊 Data Explorer", "🎯 Model Performance", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍🎓 About")
st.sidebar.info(
    "**MSc Business Analytics — AUEB**\n\n"
    "Course: Machine Learning and Content Analytics\n\n"
)

# ================================
# PAGE: HOME
# ================================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Wines Analyzed", f"{len(wine_df):,}")
    with col2:
        st.metric("Features Used", "11")
    with col3:
        st.metric("Best Model R²", "0.51")

    st.markdown("### 📖 Project Overview")
    st.markdown("""
    This dashboard predicts the quality of Portuguese Vinho Verde wines
    based on their **physicochemical properties**. The model was trained on 
    **6,497 wines** from the UCI Machine Learning Repository.

    ### 🎯 What can you do here?
    - **🔮 Predictor**: Adjust the chemistry sliders and get an instant quality prediction
    - **📊 Data Explorer**: Explore the dataset with interactive charts
    - **🎯 Model Performance**: See how our models compare
    - **ℹ️ About**: Learn about the methodology and results

    ### 🏆 Key Findings
    - **Alcohol content** is the strongest positive predictor of quality
    - **Volatile acidity** is the strongest negative predictor
    - Our tuned XGBoost model achieves **R² ≈ 0.51**, matching published benchmarks
    """)

    # Quick preview of quality distribution
    st.markdown("### 📊 Quick Look — Quality Distribution")
    fig = px.histogram(
        wine_df, x='quality', color='type',
        color_discrete_map={'red': 'crimson', 'white': 'gold'},
        title='Distribution of Wine Quality Scores by Type',
        labels={'quality': 'Quality Score', 'count': 'Number of Wines'},
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

# ================================
# PAGE: PREDICTOR
# ================================
elif page == "🔮 Predictor":
    st.markdown("### 🔮 Predict Wine Quality")
    st.markdown("Adjust the sliders on the left to describe your wine's chemistry, then click **Predict** to see the model's estimated quality!")

    # Create two columns: sliders on left, results on right
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("#### 🧪 Wine Chemistry")

        wine_type = st.selectbox("Wine Type", ["Red", "White"])
        type_encoded = 0 if wine_type == "Red" else 1

        fixed_acidity = st.slider("Fixed Acidity (g/L)", 3.8, 15.9, 7.2, 0.1,
                                  help="Non-volatile acids in the wine")
        volatile_acidity = st.slider("Volatile Acidity (g/L)", 0.08, 1.6, 0.34, 0.01,
                                     help="High values cause a vinegar taste — hurts quality")
        citric_acid = st.slider("Citric Acid (g/L)", 0.0, 1.7, 0.32, 0.01,
                                help="Adds freshness")
        residual_sugar = st.slider("Residual Sugar (g/L)", 0.6, 65.8, 5.4, 0.1,
                                   help="Sweetness after fermentation")
        chlorides = st.slider("Chlorides (g/L)", 0.009, 0.611, 0.056, 0.001,
                              help="Amount of salt")
        free_sulfur = st.slider("Free Sulfur Dioxide (mg/L)", 1.0, 289.0, 30.5, 1.0,
                                help="Anti-microbial agent")
        total_sulfur = st.slider("Total Sulfur Dioxide (mg/L)", 6.0, 440.0, 115.7, 1.0,
                                 help="Total SO₂")
        density = st.slider("Density (g/cm³)", 0.987, 1.039, 0.995, 0.0001, "%.4f",
                            help="Related to alcohol & sugar content")
        pH = st.slider("pH", 2.72, 4.01, 3.22, 0.01,
                       help="Acidity level (lower = more acidic)")
        sulphates = st.slider("Sulphates (g/L)", 0.22, 2.0, 0.53, 0.01,
                              help="Antimicrobial preservative")
        alcohol = st.slider("Alcohol (%)", 8.0, 14.9, 10.5, 0.1,
                            help="🍷 The strongest positive predictor!")

    with col_result:
        st.markdown("#### 🎯 Prediction Results")

        # Build feature array
        input_data = pd.DataFrame({
            'fixed acidity': [fixed_acidity],
            'volatile acidity': [volatile_acidity],
            'citric acid': [citric_acid],
            'residual sugar': [residual_sugar],
            'chlorides': [chlorides],
            'free sulfur dioxide': [free_sulfur],
            'total sulfur dioxide': [total_sulfur],
            'density': [density],
            'pH': [pH],
            'sulphates': [sulphates],
            'alcohol': [alcohol],
            'type_encoded': [type_encoded]
        })

        # Reorder to match model's feature order
        input_data = input_data[feature_names]

        # Predict
        reg_prediction = regressor.predict(input_data)[0]
        clf_prediction_enc = classifier.predict(input_data)[0]
        clf_prediction = label_encoder.inverse_transform([clf_prediction_enc])[0]
        clf_probs = classifier.predict_proba(input_data)[0]

        # Display prediction
        st.markdown(f"""
        <div style="background-color:#F8F0E3; padding:2rem; border-radius:15px; text-align:center;">
            <h2 style="color:#722F37; margin:0;">Predicted Quality</h2>
            <h1 style="color:#722F37; font-size:4rem; margin:0.5rem 0;">{reg_prediction:.2f} / 10</h1>
            <h3 style="color:#722F37; margin:0;">Category: <b>{clf_prediction}</b></h3>
        </div>
        """, unsafe_allow_html=True)

        # Emoji reaction
        if reg_prediction >= 7:
            st.balloons()
            st.success("🌟 **Excellent wine!** This one deserves a special occasion.")
        elif reg_prediction >= 6:
            st.info("👍 **Good quality wine.** Perfectly enjoyable!")
        elif reg_prediction >= 5:
            st.warning("😐 **Average wine.** Nothing to write home about.")
        else:
            st.error("⚠️ **Below average quality.** Would need improvements.")

        # Show class probabilities
        st.markdown("#### 📊 Class Probabilities")
        prob_df = pd.DataFrame({
            'Category': label_encoder.classes_,
            'Probability': clf_probs
        }).sort_values('Probability', ascending=True)

        fig = px.bar(
            prob_df, x='Probability', y='Category',
            orientation='h',
            color='Probability',
            color_continuous_scale='RdYlGn',
            text=prob_df['Probability'].apply(lambda x: f'{x:.1%}')
        )
        fig.update_traces(textposition='inside')
        fig.update_layout(showlegend=False, height=250)
        st.plotly_chart(fig, use_container_width=True)

    # SHAP explanation
    st.markdown("---")
    st.markdown("### 🔍 Why did the model predict this?")
    st.markdown("The chart below shows how each feature **pushed the prediction up or down** from the average.")

    with st.spinner("Computing SHAP explanation..."):
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(input_data)

        # Build a waterfall-like chart with Plotly
        base_value = explainer.expected_value
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': input_data.iloc[0].values,
            'SHAP': shap_values[0]
        }).sort_values('SHAP', key=abs, ascending=True)

        colors = ['crimson' if v < 0 else 'seagreen' for v in shap_df['SHAP']]
        fig = go.Figure(go.Bar(
            x=shap_df['SHAP'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=colors,
            text=shap_df.apply(lambda r: f"{r['Feature']}={r['Value']:.2f} → {r['SHAP']:+.3f}", axis=1),
            textposition='outside'
        ))
        fig.update_layout(
            title=f"Feature contributions (base value: {base_value:.2f})",
            xaxis_title="SHAP value (impact on prediction)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **How to read this chart:**
        - 🟢 **Green bars** pushed the prediction **UP** (better quality)
        - 🔴 **Red bars** pushed the prediction **DOWN** (worse quality)
        - The longer the bar, the bigger the impact
        """)

# ================================
# PAGE: DATA EXPLORER
# ================================
elif page == "📊 Data Explorer":
    st.markdown("### 📊 Explore the Wine Dataset")

    # Filter by wine type
    wine_type_filter = st.multiselect(
        "Filter by wine type:",
        options=['red', 'white'],
        default=['red', 'white']
    )
    filtered_df = wine_df[wine_df['type'].isin(wine_type_filter)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wines", f"{len(filtered_df):,}")
    col2.metric("Avg Alcohol", f"{filtered_df['alcohol'].mean():.2f}%")
    col3.metric("Avg pH", f"{filtered_df['pH'].mean():.2f}")
    col4.metric("Avg Quality", f"{filtered_df['quality'].mean():.2f}")

    st.markdown("#### 🎨 Interactive Feature Explorer")
    numeric_cols = [c for c in filtered_df.columns if c not in ['type', 'type_encoded']]

    col1, col2 = st.columns(2)
    x_feature = col1.selectbox("X-axis:", numeric_cols, index=numeric_cols.index('alcohol'))
    y_feature = col2.selectbox("Y-axis:", numeric_cols, index=numeric_cols.index('quality'))

    fig = px.scatter(
        filtered_df, x=x_feature, y=y_feature, color='type',
        color_discrete_map={'red': 'crimson', 'white': 'gold'},
        opacity=0.5,
        trendline='ols',
        title=f"{x_feature} vs {y_feature}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔗 Feature Correlation Heatmap")
    corr = filtered_df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto='.2f', color_continuous_scale='RdBu_r',
        aspect='auto', title='Correlation Matrix'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    with st.expander("📋 View Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)

# ================================
# PAGE: MODEL PERFORMANCE
# ================================
elif page == "🎯 Model Performance":
    st.markdown("### 🎯 Model Performance Comparison")

    results = pd.DataFrame({
        'Model': ['Baseline (Mean)', 'Linear Regression', 'Random Forest', 'XGBoost', 'Tuned XGBoost'],
        'RMSE': [0.874, 0.728, 0.615, 0.610, 0.619],
        'MAE': [0.686, 0.568, 0.447, 0.448, 0.457],
        'R²': [0.000, 0.306, 0.505, 0.513, 0.498]
    })

    st.dataframe(results.style.highlight_max(subset=['R²'], color='lightgreen')
                              .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                 use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(results, x='Model', y='RMSE', color='RMSE',
                     color_continuous_scale='RdYlGn_r', title='RMSE (lower is better)')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(results, x='Model', y='R²', color='R²',
                     color_continuous_scale='RdYlGn', title='R² (higher is better)')
        st.plotly_chart(fig, use_container_width=True)

    st.info("💡 **Winner**: XGBoost (untuned) with R² = 0.513. Interestingly, hyperparameter tuning gave slightly worse results — the defaults were already excellent!")

# ================================
# PAGE: ABOUT
# ================================
elif page == "ℹ️ About":
    st.markdown("### ℹ️ About This Project")
    st.markdown("""
    #### 📖 Project
    This is my final project for the course **Machine Learning and Content Analytics**  
    in the **MSc Business Analytics** program at **Athens University of Economics and Business (AUEB)**.

    #### 🎯 Business Question
    *Can we build a machine learning system that predicts wine quality from lab measurements
    accurately enough to support quality control in wineries, reducing the reliance on subjective human tasting panels?*

    #### 📊 Dataset
    - **Source**: UCI Machine Learning Repository
    - **Reference**: Cortez, P. et al. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems.
    - **Size**: 6,497 wines (1,599 red + 4,898 white) from the Vinho Verde region of Portugal
    - **Features**: 11 physicochemical measurements per wine

    #### 🛠️ Tech Stack
    - **Language**: Python 3
    - **ML**: scikit-learn, XGBoost
    - **Explainability**: SHAP
    - **Dashboard**: Streamlit
    - **Deployment**: Streamlit Cloud

    #### 📚 References
    - Cortez et al. (2009) — Original wine dataset paper
    - Chen & Guestrin (2016) — XGBoost paper
    - Lundberg & Lee (2017) — SHAP paper
    """)

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888;'>"
    "🍷 Made with Streamlit • MSc Business Analytics @ AUEB • 2026"
    "</p>",
    unsafe_allow_html=True
)
