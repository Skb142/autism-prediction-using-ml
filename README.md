<h1 align="center">🧠 Autism Spectrum Prediction using Machine Learning</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/ML-XGBoost | LGBM | CatBoost-green?style=flat-square" />
</p>

<p align="center">
  A complete machine learning pipeline for predicting Autism Spectrum Disorder (ASD) risk  
  using modern ML models, feature engineering, SMOTE balancing, and SHAP explainability.  
  Designed as a research + portfolio project.

---

## 🚀 **Live Demo (Open Notebook Online — No Setup Needed!)**

Click below to run the notebook on **Binder + Voilà**:

👉 **Launch Interactive Notebook**  
https://mybinder.org/v2/gh/Skb142/autism-prediction-using-ml/HEAD?urlpath=voila%2Frender%2Fnotebooks%2Fautism_prediction_template%20(2).ipynb

> Runs entirely in the browser — no installation required.

---

## 📘 **About the Project**

This project explores whether Autism Spectrum Disorder (ASD) can be predicted using machine learning models trained on behavioral, demographic, and screening questionnaire features.

### ✨ Key Highlights
- 🔍 **Data preprocessing** and feature cleaning  
- 🤖 **Machine Learning Models**  
  - XGBoost  
  - LightGBM  
  - CatBoost  
  - Random Forest (baseline)  
- ⚖️ **SMOTE** oversampling to fix class imbalance  
- 📊 **Performance metrics**: accuracy, F1 score, ROC-AUC  
- 🧩 **Explainability using SHAP** plots  
- 📝 A detailed research paper included  

---

## 📁 **Project Structure**
autism-prediction-using-ml/
│
├── notebooks/ # Jupyter notebooks & research paper
│ ├── autism_prediction_template (2).ipynb
│ └── AutismResearch_final.docx
│
├── src/ # Source code (optional future use)
│ └── README.md
│
├── models/ # Trained models (not uploaded)
│ └── README.md
│
├── examples/ # Example input files
│ └── sample_input.csv
│
├── requirements.txt # Dependencies
└── README.md # You're reading it!

---

## 🧪 **Technologies & Tools Used**

| Category | Tools/Packages |
|---------|----------------|
| ML Models | XGBoost, LightGBM, CatBoost, RandomForest |
| Data Handling | Pandas, NumPy |
| Balancing | SMOTE (Imbalanced-Learn) |
| Explainability | SHAP |
| Notebook | Jupyter, Voilà |
| Visualization | Matplotlib, Seaborn |
| Deployment | Binder |

---

## 🧠 **Workflow Overview**

1. **Load dataset**  
2. **Data preprocessing**  
   - Missing values  
   - Encoding categorical features  
   - Scaling numerical features  
3. **Train-Test Split**  
4. **Model Training**  
5. **SMOTE for imbalance handling**  
6. **Evaluation**  
7. **SHAP Explainability**  
8. **Saving notebook results for research documentation**

---

## 📄 **Research Paper**

📘 A detailed research paper explaining methodology, models, results, and limitations  
is included inside the **`/notebooks/`** folder.

---

## 🚀 **Future Improvements**

- Deploy as a **Streamlit web app**  
- Add proper **inference pipeline** in `src/`  
- Build an **API using FastAPI**  
- Extend dataset with more clinical features  

---

## 🙌 **Contributors**

👤 **Sahil Kumar Behera**  
Final Year B.Tech CSE  
KIIT University  

If you'd like a more advanced deployment (Streamlit / FastAPI / HuggingFace), feel free to ask!

---

## ⭐ **Support**

If you like this project, consider ⭐ starring the repo — it helps a lot!
