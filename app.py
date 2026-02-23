import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(page_title="HR Analytics Dashboard",
                   page_icon="📊",
                   layout="wide")

# Load model
cm = pickle.load(open("confusion_matrix.pkl", "rb"))

model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Sidebar Navigation
st.sidebar.title("Navigation")
st.sidebar.write("Model: Best of Logistic & Random Forest")

page = st.sidebar.radio("Go to", ["🏠 Dashboard", "🔮 Prediction"])

# ---------------- DASHBOARD PAGE ---------------- #
if page == "🏠 Dashboard":

    st.title("📊 HR Analytics Dashboard")

    df = pd.read_csv("data.csv")

    total_employees = len(df)
    attrition_count = len(df[df["Attrition"] == "Yes"])
    retention_rate = round((1 - attrition_count/total_employees) * 100, 2)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Employees", total_employees)
    col2.metric("Employees Left", attrition_count)
    col3.metric("Retention Rate", f"{retention_rate}%")

    st.markdown("---")

    st.subheader("Attrition Distribution")
    fig = px.pie(df, names="Attrition", title="Attrition Breakdown")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attrition by Department")
    fig2 = px.histogram(df, x="Department", color="Attrition",
                        barmode="group")
    st.plotly_chart(fig2, use_container_width=True)
    
 

# ---------------- PREDICTION PAGE ---------------- #
elif page == "🔮 Prediction":

    st.title("🔮 Employee Attrition Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 60)
        years_at_company = st.slider("Years at Company", 0, 40)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4)

    with col2:
        monthly_income = st.number_input("Monthly Income", min_value=1000)
        work_life_balance = st.slider("Work Life Balance (1-4)", 1, 4)
        distance_from_home = st.slider("Distance From Home", 1, 30)

    st.markdown("---")

    if st.button("Predict Now"):

        input_data = pd.DataFrame(columns=features)
        input_data.loc[0] = 0

        for col in input_data.columns:
            if col == "Age":
                input_data[col] = age
            elif col == "YearsAtCompany":
                input_data[col] = years_at_company
            elif col == "MonthlyIncome":
                input_data[col] = monthly_income
            elif col == "JobSatisfaction":
                input_data[col] = job_satisfaction
            elif col == "WorkLifeBalance":
                input_data[col] = work_life_balance
            elif col == "DistanceFromHome":
                input_data[col] = distance_from_home

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Prediction Result")

        colA, colB = st.columns(2)
        colA.metric("Probability of Leaving",
                    f"{round(probability*100,2)}%")

        if prediction[0] == 1:
            colB.error("High Risk")
        else:
            colB.success("Low Risk")

        st.markdown("---")

        st.subheader("Top Important Features")

        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=features)
        top_features = feature_importance.sort_values(
            ascending=False)[:10]

        fig3 = px.bar(top_features,
                      x=top_features.values,
                      y=top_features.index,
                      orientation='h',
                      title="Feature Importance")

        st.plotly_chart(fig3, use_container_width=True)

