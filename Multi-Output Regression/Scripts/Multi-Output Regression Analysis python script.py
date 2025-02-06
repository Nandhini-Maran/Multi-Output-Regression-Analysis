#!/usr/bin/env python
# coding: utf-8

# In[172]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns  


# # IMPORTING THE DATASET

# In[173]:


df=pd.read_csv('Life Expectancy Data.csv')


#    # OVER VIEW OF DATASET  

# In[174]:


df.head()


# In[175]:


df.tail()


# In[176]:


df.shape


# In[177]:


df.columns


# In[178]:


df.columns = df.columns.str.strip()


# In[179]:


print(df.columns.tolist())


# In[180]:


for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        print(f"{col} is Numeric")
    elif pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
        print(f"{col} is Categorical")


# In[181]:


df.dtypes


# In[182]:


df.nunique()


# In[183]:


df.describe()


# In[184]:


df.info()


# In[185]:


df.isnull().sum()


#  # FILLING NULL VALUES

# In[186]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

for column in numerical_columns:
    median_value = df[column].median()  
    df[column].fillna(median_value, inplace=True) 


# In[187]:


df.isnull().sum()


#   # ANALYSING THE DATA 

# In[188]:


plt.figure(figsize=(10,5))
df[numerical_columns].boxplot(rot=90,fontsize=10)
plt.title("Box Plot for Numerical Features",fontsize=10)
plt.show


# In[189]:


df.corr(numeric_only=True)


# In[190]:


plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features', fontsize=14)
plt.show()


#   # REMOVED LESS IMPACTFUL COLUMNS BY USING CORRELATION AND VIF

# In[191]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[192]:


X = df[numerical_columns] 
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)


# In[193]:


df=df.drop(columns=['infant deaths','thinness 5-9 years','GDP'],axis=1)
df.head()


# In[194]:


df.corr(numeric_only=True)
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr(numeric_only=True) 
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features', fontsize=14)
plt.show()


# In[195]:


numerical_columns=df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns


#   # DISTRIBUTION OF COLUMNS 

# In[196]:


num_columns = len(numerical_columns)
n_cols = 4 
n_rows = (num_columns // n_cols) + (num_columns % n_cols > 0)


plt.figure(figsize=(20, n_rows * 5))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=df[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[197]:


df_encoded = df.copy()


#   # PREPROCESSING THE DATA

# In[198]:


from sklearn.preprocessing import LabelEncoder


# In[199]:


label_encoder=LabelEncoder()


# In[200]:


df_encoded['Country_encoded'] = label_encoder.fit_transform(df_encoded['Country'])


# In[201]:


df_encoded['status_encoded'] = label_encoder.fit_transform(df_encoded['Status'])


# In[202]:


df_encoded = df_encoded.drop(columns=['Country', 'Status'])


# In[203]:


df_encoded


# In[204]:


X_pca=df_encoded.drop(columns=['Life expectancy','BMI'])


# In[205]:


from sklearn.preprocessing import StandardScaler


# In[206]:


scalar=StandardScaler()


# In[207]:


scaled_data=scalar.fit_transform(X_pca)


# In[208]:


scaled_data.shape


#   # APPLYING PCA FOR REDUCING THE DIMENSIONS 

# In[209]:


from sklearn.decomposition import PCA


# In[210]:


pca=PCA(0.90)


# In[211]:


scaled_pca=pca.fit_transform(scaled_data)
scaled_pca


# In[212]:


explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
print("Cumulative Explained Variance:", explained_variance.cumsum())


# In[213]:


import matplotlib.pyplot as plt

plt.scatter(scaled_pca[:, 0], scaled_pca[:, 1], alpha=0.5)
plt.title("PCA Scatter Plot")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()


# In[214]:


for i in range(scaled_pca.shape[1]):
    df_encoded[f'PC{i+1}'] = scaled_pca[:, i]

print(df_encoded.head())


# In[215]:


from sklearn.model_selection import train_test_split


# In[216]:


x = df_encoded[[f'PC{i+1}' for i in range(scaled_pca.shape[1])]] 
y = df_encoded[['Life expectancy', 'BMI']]  


# In[217]:


x


# In[218]:


y


# In[219]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[220]:


y_train


#   # APPLYING THE PREPROCESED PCA FEATURES TO THE MODEL

# In[221]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[222]:


lr.fit(X_train,y_train)


# In[223]:


Y_pred_lr=lr.predict(X_test)


#   # VALUATING THE MODEL PERFORMANCE 

# In[224]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error


# In[225]:


print(r2_score(y_test,Y_pred_lr,multioutput='raw_values'))
print(mean_absolute_error(y_test,Y_pred_lr,multioutput='raw_values'))
print(mean_squared_error(y_test,Y_pred_lr,multioutput='raw_values'))
print(np.sqrt(mean_squared_error(y_test,Y_pred_lr,multioutput='raw_values')))


#   # NORMALIZING THE DISTRIBUTION OF THE COLUMNS FOR BETTER PERFORMANCE 

# In[226]:


from scipy.stats import boxcox
from scipy.stats import zscore


# In[227]:


transformed_df = df.copy()


# In[228]:


transformed_df


# In[229]:


transformed_numerical_columns=transformed_df.select_dtypes(include=['float64', 'int64']).columns
transformed_numerical_columns


# In[230]:


for col in transformed_numerical_columns:
    if (df[col] <= 0).any():
        transformed_df[col] = np.log1p(df[col])  
    else:
        transformed_df[col], _ = boxcox(df[col]) 
    

    transformed_df[col] = zscore(transformed_df[col])


# In[231]:


transformed_df


#   # NORMALIZED DISTRIBUTION OF THE DATASET 

# In[232]:


num_columns = len(transformed_numerical_columns)
n_cols = 4 
n_rows = (num_columns // n_cols) + (num_columns % n_cols > 0)


plt.figure(figsize=(20, n_rows * 5))

for i, col in enumerate(transformed_numerical_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=transformed_df[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[233]:


transformed_df_encoded = transformed_df.copy()


# In[234]:


transformed_df_encoded


#   # PREPROCESSING THE NORMALIZED DATA'S 

# In[235]:


transformed_df_encoded['Country_encoded'] = label_encoder.fit_transform(transformed_df_encoded['Country'])


# In[236]:


transformed_df_encoded['status_encoded'] = label_encoder.fit_transform(transformed_df_encoded['Status'])


# In[237]:


transformed_df_encoded = transformed_df_encoded.drop(columns=['Country', 'Status'])


# In[238]:


transformed_df_encoded


# In[239]:


X_pca=transformed_df_encoded.drop(columns=['Life expectancy','BMI'])


#   # PCA FOR NORMALIZED DATA

# In[240]:


scaled_data=scalar.fit_transform(X_pca)


# In[241]:


pca=PCA(0.90)


# In[242]:


scaled_pca=pca.fit_transform(scaled_data)
scaled_pca


# In[243]:


for i in range(scaled_pca.shape[1]):
    df_encoded[f'PC{i+1}'] = scaled_pca[:, i]

print(df_encoded.head())


# In[244]:


x = df_encoded[[f'PC{i+1}' for i in range(scaled_pca.shape[1])]] 
y = df_encoded[['Life expectancy', 'BMI']]  


# In[245]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)


#   # APPLYING NORMALIZED PCA FOR DIFFERENT MODELS

# In[246]:


from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import AdaBoostRegressor


# In[247]:


models = {
    "LinearRegression": LinearRegression(),
    "SVR": MultiOutputRegressor(SVR()),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "AdaBoost": MultiOutputRegressor(AdaBoostRegressor())}


# In[248]:


models


# In[249]:


Rs = []    
Mae = []   
Mse = []
Rmse = []


# In[250]:


for model_name, model in models.items():
       
     model.fit(X_train, y_train)
     y_pred = model.predict(X_test)
        
     rs = r2_score(y_test, y_pred, multioutput='raw_values')
     mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
     mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
     rmse = np.sqrt(mse)

     Rs.append(rs)
     Mae.append(mae)
     Mse.append(mse)
     Rmse.append(rmse)
        
       
     print(f"{model_name} - R²: {rs}")
     print(f"{model_name} - MAE: {mae}")
     print(f"{model_name} - MSE: {mse}")
     print(f"{model_name} - RMSE: {rmse}\n")
   


# In[251]:


summary_df = pd.DataFrame({
    'Model': list(models.keys()),
    'R2_Life': [score[0] for score in Rs],
    'R2_BMI': [score[1] for score in Rs],
    'MAE_Life': [score[0] for score in Mae],
    'MAE_BMI': [score[1] for score in Mae],
    'MSE_Life' :[score[0] for score in Mse],
    'MSE_BMI' :[score[1] for score in Mse],
    'RMSE_Life': [score[0] for score in Rmse],
    'RMSE_BMI': [score[1] for score in Rmse]
})

summary_df


#   # HYPERPERAMETER TUNING FOR EACH MODEL FOR BETTER RESULTS

# In[252]:


from sklearn.model_selection import GridSearchCV


# In[253]:


param_grids = {
    "Linear Regression": {},  
    "Decision_Tree": {
        'estimator__max_depth': [None, 10, 20, 30],
        'estimator__min_samples_split': [2, 5, 10]
    },
    "Random_Forest": {
        'estimator__n_estimators': [10, 50, 100],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5]
    },
    "SVR": {
        'estimator__kernel': ['linear', 'rbf'],
        'estimator__C': [0.1, 1, 10],
        'estimator__gamma': ['scale', 'auto']
    },
    "AdaBoost": {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.01, 0.1, 1.0]
    }
}


# In[254]:


models = {
    "Linear Regression": LinearRegression(),
    "Decision_Tree": DecisionTreeRegressor(),
    "Random_Forest": RandomForestRegressor(),
    "SVR": SVR(),
    "AdaBoost": AdaBoostRegressor()}


# In[255]:


best_models = {}
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    if model_name == "Linear Regression": 
        best_models[model_name] = MultiOutputRegressor(model).fit(X_train, y_train)
    else:
        grid_search = GridSearchCV(
            estimator=MultiOutputRegressor(model),
            param_grid=param_grids[model_name],
            scoring='r2',
            cv=3,
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")


# In[256]:


RS = []    
MAE = []   
MSE=[]
RMSE = []


# In[257]:


for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    rs = r2_score(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))  # Correct RMSE calculation

    RS.append(rs)
    MAE.append(mae)
    MSE.append(mse)
    RMSE.append(rmse)

    print(f"{model_name} - RS: {rs}")
    print(f"{model_name} - MAE: {mae}")
    print(f"{model_name} - MSE: {mse}")
    print(f"{model_name} - RMSE: {rmse}\n")


#   # TUNED MODEL PERFORMANCE  

# In[258]:


summary_df = pd.DataFrame({
    'Model': list(best_models.keys()),
    'R2_Life': [score[0] for score in RS],
    'R2_BMI': [score[1] for score in RS],
    'MAE_Life': [score[0] for score in MAE],
    'MAE_BMI': [score[1] for score in MAE],
    'MSE_Life' :[score[0] for score in MSE],
    'MSE_BMI' :[score[1] for score in MSE],
    'RMSE_Life': [score[0] for score in RMSE],
    'RMSE_BMI': [score[1] for score in RMSE]
})

summary_df


#   # PREDICTED VALUES FOR EACH MODELS 

# In[262]:


for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    
   
    for i, target_name in enumerate(y_test.columns):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test[target_name], y_pred[:, i], alpha=0.6, label='Predicted')
        plt.plot(
            [y_test[target_name].min(), y_test[target_name].max()],
            [y_test[target_name].min(), y_test[target_name].max()],
            color='red', linestyle='--', label='Perfect Prediction'
        )
        plt.title(f'{model_name} - Predicted Values - {target_name}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.show()


#   # ERROR DISTRIBUTION FOR EACH MODELS 

# In[260]:


for model_name, model in best_models.items():

    y_pred = model.predict(X_test)
    
   
    for i, target_name in enumerate(y_test.columns):
        errors = y_test[target_name] - y_pred[:, i]
        plt.figure(figsize=(8, 6))
        plt.hist(errors, bins=30, color='purple', alpha=0.7)
        plt.title(f'{model_name} - Error Distribution - {target_name}')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.show()


# # FOR OUR REGRESSION ANALYSIS RANDOM FOREST PERFORMS BETTER ON BOTH WITH AND WITHOUT HYPERTUNNING FOR BOTH TARGET VARIIABLES  

# #   WITHOUT HYPERTUNNING :-
# #   Random_Forest - R_SCORE: [0.89041872 0.53482908]
# #   WITH HYPERTUNNIG:-
# #   Random_Forest - R_SCORE: [0.88950839 0.52331455]
