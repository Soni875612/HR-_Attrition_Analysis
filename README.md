# HR-_Attrition_Analysis
# 📊 HR Attrition Analytics & Prediction System

An end-to-end Machine Learning application that analyzes employee attrition trends and predicts employee churn risk using classification models. The system integrates predictive analytics with an interactive Streamlit dashboard to support data-driven HR decision making.

---

## 🚀 Project Overview

Employee attrition significantly impacts organizational productivity and cost. This project builds a predictive HR analytics system that:

• Identifies employees at high risk of leaving  
• Analyzes attrition distribution across departments  
• Calculates retention metrics  
• Provides probability-based risk prediction  
• Visualizes model feature importance  

The solution combines machine learning with business intelligence reporting.

---

## 🧠 Machine Learning Pipeline

Data → Cleaning → Encoding → Train-Test Split → Model Training → Model Selection → Deployment

### Models Used:
- Logistic Regression  
- Random Forest Classifier  

✔ Automatic best model selection  
✔ 80% Training / 20% Testing  
✔ Confusion Matrix evaluation  

---

## 📊 Dashboard Features

### 🏠 HR Analytics Dashboard
- Total Employees
- Employees Left
- Retention Rate
- Attrition Distribution (Pie Chart)
- Attrition by Department (Bar Chart)

### 🔮 Prediction Module
User inputs:
- Age
- Years at Company
- Monthly Income
- Job Satisfaction
- Work-Life Balance
- Distance from Home

System Outputs:
- Probability of Leaving (%)
- Risk Level (High / Low)
- Top Important Features

---

## 📂 Project Structure

HR-Attrition-Analysis/
│
├── app.py
├── train_model.py
├── data.csv
├── model.pkl
├── confusion_matrix.pkl
├── features.pkl
├── requirements.txt
└── README.md

---

## 📊 Dataset

The dataset contains employee information including demographic details, income, job satisfaction, department, and attrition status.

If included in the repository, download from:

https://github.com/Soni875612/HR-Attrition-Analysis/raw/main/data.csv



---

## ⚙️ Installation & Execution

1️⃣ Clone Repository

git clone https://github.com/Soni875612e/HR-Attrition-Analysis.git  
cd HR-Attrition-Analysis  

2️⃣ Install Dependencies

pip install -r requirements.txt  

3️⃣ Run Application

streamlit run app.py  

---

## 🏗 Tech Stack

• Python  
• Pandas  
• NumPy  
• Scikit-learn  
• Streamlit  
• Plotly  

---
## project screenshot
<img width="1919" height="907" alt="Screenshot 2026-02-16 200910" src="https://github.com/user-attachments/assets/d3fa5b90-8952-47fb-acf2-c8ed9c4c668b" />
<img width="1919" height="885" alt="image" src="https://github.com/user-attachments/assets/3b6287d8-0b75-48e0-a73b-dc2f09c7e929" />

## 📈 Business Impact

• Helps HR teams proactively manage employee retention  
• Reduces unexpected turnover  
• Supports strategic workforce planning  
• Enables explainable AI-based HR decisions  

---

## 🔮 Future Improvements

• SHAP explainability integration  
• Cloud deployment  
• Advanced ensemble models  
• Model comparison dashboard  

---

## 👨‍💻 Author

Soni  
Machine Learning & Data Analytics Enthusiast

