# Multi-Output Regression Analysis on Life Expectancy and BMI

## 📌 Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Models Implemented](#models-implemented)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Results](#results)
- [Visualization](#visualization)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)

---

## 📖 Project Description
This project focuses on **multi-output regression analysis** to predict two key health indicators:
- **Life Expectancy**
- **BMI (Body Mass Index)**

We apply various regression models to evaluate the best approach for accurate prediction using different techniques, including:
- Feature engineering and data preprocessing.
- Principal Component Analysis (PCA) for dimensionality reduction.
- Model evaluation using performance metrics like R², MAE, MSE, and RMSE.
- Hyperparameter tuning for optimization.

---

## 📊 Dataset
- **Dataset Name:** `Life Expectancy Data.csv`
- **Source:** [WHO, United Nations](https://www.kaggle.com/kumarajarshi/life-expectancy-who)
- **Features:** The dataset contains health, economic, and demographic indicators for various countries.
- **Data Preprocessing:**
  - Missing values were handled using median imputation.
  - Categorical variables were label-encoded.
  - Feature selection was done using correlation analysis and Variance Inflation Factor (VIF).
  - PCA was applied for dimensionality reduction.

---

## 🛠 Technologies Used
- **Programming Language:** Python
- **Libraries Used:**
  - 📊 `pandas`, `numpy` → Data manipulation
  - 🎨 `matplotlib`, `seaborn` → Data visualization
  - 🤖 `scikit-learn` → Machine learning models & preprocessing
  - 📈 `statsmodels` → Statistical analysis
  - 🔧 `GridSearchCV` → Hyperparameter tuning

---

## 🔄 Project Workflow
### **1️⃣ Data Exploration**
- Handled missing values using median imputation.
- Checked for categorical and numerical features.
- Performed correlation analysis and visualized feature distributions.

### **2️⃣ Feature Engineering**
- Applied **Label Encoding** for categorical variables.
- Used **Principal Component Analysis (PCA)** to reduce dimensions.
- Standardized numerical features using **Z-score normalization**.

### **3️⃣ Model Development**
- Implemented multiple regression models.
- Compared models using various evaluation metrics.

### **4️⃣ Hyperparameter Tuning**
- Used **GridSearchCV** to optimize parameters.

### **5️⃣ Model Evaluation**
- Analyzed model performance using:
  - R² Score
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

---

## 🤖 Models Implemented
| Model | R² Score (Life Expectancy) | R² Score (BMI) |
|--------|--------------------------|---------------|
| **Linear Regression** | 0.79 | 0.42 |
| **Decision Tree Regressor** | 0.83 | 0.48 |
| **Random Forest Regressor** | **0.89** | **0.53** |
| **Support Vector Regressor (SVR)** | 0.76 | 0.39 |
| **AdaBoost Regressor** | 0.81 | 0.45 |

✅ **Random Forest performed the best for both targets!**

---

## 🎯 Hyperparameter Tuning
**Optimized Parameters Used:**
- **Decision Tree:**
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]
- **Random Forest:**
  - `n_estimators`: [10, 50, 100]
  - `max_depth`: [None, 10, 20]
- **SVR:**
  - `kernel`: [‘linear’, ‘rbf’]
  - `C`: [0.1, 1, 10]
- **AdaBoost:**
  - `n_estimators`: [50, 100, 200]
  - `learning_rate`: [0.01, 0.1, 1.0]

---

## 📈 Results
### **Without Hyperparameter Tuning**
- **Random Forest**:
  - **R² for Life Expectancy:** 0.8904
  - **R² for BMI:** 0.5348

### **With Hyperparameter Tuning**
- **Random Forest (Tuned)**
  - **R² for Life Expectancy:** 0.8895
  - **R² for BMI:** 0.5233

---

## 📊 Visualization
📌 **Correlation Heatmap**
![Correlation Matrix](![image](https://github.com/user-attachments/assets/1880c45d-02ef-4749-a206-ee8af7a0fe49)

)

📌 **Predicted Values**
![Predicted](  
![image](https://github.com/user-attachments/assets/b66cbf48-598b-451d-b509-6294cca53f3a),![image](https://github.com/user-attachments/assets/0bc9aa13-842a-4a38-89d5-a2f22c68f4e2)

)

📌 **Error Distribution**
![Error Distribution](![image](https://github.com/user-attachments/assets/09341200-67e2-4aad-b61b-57ab6b78f22f),![image](https://github.com/user-attachments/assets/abab4ab5-dc10-494a-80e2-531cb32d64df)

) 
---

## 🔍 Conclusion
- **Random Forest Regressor** is the best model for predicting both Life Expectancy and BMI.
- PCA effectively reduced dimensionality while retaining key information.
- Feature engineering and normalization improved model performance.

---

## How to Run
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Multiclass-Classification-Project.git
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the classification script**:
   ```bash
   python Multi-Output Regression Analysis python script.PY
   ```
4. **Check results and visualizations** in the console.
