# Heart Disease Diagnosis with Artificial Neural Networks (ANN)

This project uses a public dataset from the Cleveland Clinic Foundation to diagnose heart disease based on clinical variables. The focus is to apply artificial neural networks (ANN) to predict disease risk, along with exploratory data analysis and hyperparameter tuning.

---

## 📁 Dataset

The dataset includes 14 variables such as age, sex, blood pressure, cholesterol, chest pain type, etc. The target variable (`Target`) indicates whether heart disease is present (1 = yes; 0 = no).

Sample metadata:

| Column   | Description                                         | Feature Type                  |
|----------|-----------------------------------------------------|-------------------------------|
| Age      | Age in years                                        | Numerical                     |
| Sex      | (1 = male; 0 = female)                              | Categorical                   |
| CP       | Chest pain type (0–4)                               | Categorical                   |
| Trestbpd | Resting blood pressure                              | Numerical                     |
| Chol     | Serum cholesterol in mg/dl                          | Numerical                     |
| FBS      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)| Categorical                  |
| ...      | ...                                                 | ...                           |
| Target   | Heart disease (1 = yes; 0 = no)                     | Target                        |

---

## 🔍 Project Steps

- **Exploratory Data Analysis (EDA)**:
  - Check for missing or duplicated values
  - Convert numerical codes to categorical variables
  - Generate automated reports via `pandas-profiling` and `sweetviz`
  - Create visualizations like `pairplot`

- **Data Preprocessing**:
  - Train-test split
  - Scale features using `StandardScaler`
  - Encode categorical variables using `OneHotEncoder`

- **Modeling with ANN (Keras/TensorFlow)**:
  - Architecture with 2 hidden layers
  - Early stopping using AUC as evaluation metric
  - Training for 50 epochs
  - Evaluation based on AUC (train vs test)

- **Hyperparameter Tuning**:
  - Generate 50 ANN configurations
  - Create a result table with AUC scores
  - Analyze performance using line plots and scatter plots

- **New patient prediction**:
  - Create a DataFrame with new patient data
  - Apply the same preprocessing
  - Predict heart disease probability using best model

---

## 🔁 Reproducibility

Training neural networks involves stochastic components such as random weight initialization, data shuffling and dropout.

To ensure reproducibility of the reported results — including metrics, plots and comparisons across executions — a fixed random seed was used during model training.

---

## 🧠 Technologies

- Python 3.x
- Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- TensorFlow (Keras)
- Pandas Profiling, Sweetviz

---

## ✅ Requirements

```txt
matplotlib==3.2.2
numpy==1.19.5
pandas==1.2.5
scikit-learn==0.24.0
scipy==1.7.2
seaborn==0.10.1
tensorflow==2.4.1
pandas-profiling==3.1.0
sweetviz==1.0b6
