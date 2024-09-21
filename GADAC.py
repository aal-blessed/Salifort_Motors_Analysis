#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Step 1: Importing necessary libraries
import pandas as pd

# Step 2: Reading the CSV file
file_path = r'C:\Users\TheBlessed\Documents\GADAC\gada_cap.csv'
data = pd.read_csv(file_path)

# Step 3: Display the first 5 rows of the dataset
data.head()


# In[2]:


# Step 4: Display basic information about the dataset
data.info()


# In[3]:


# Step 5: Display descriptive statistics of numerical columns
data.describe()


# In[4]:


# Step 6: Importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Step 7: Plotting pairplot to visualize relationships
sns.pairplot(data[['last_evaluation', 'tenure', 'number_project', 'left']])
plt.show()


# In[5]:


# Step 8: Calculate correlation matrix
corr_matrix = data.corr()

# Step 9: Plotting the heatmap to visualize correlations
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[6]:


# Step 10: Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

# Encoding the 'department' column
label_encoder = LabelEncoder()
data['department'] = label_encoder.fit_transform(data['department'])


# In[7]:


# Step 11: Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

# Features and target variable
X = data.drop('left', axis=1)
y = data['left']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 12: Building the Decision Tree model
from sklearn.tree import DecisionTreeClassifier

# Initialize the model
model = DecisionTreeClassifier(random_state=42)

# Fit the model
model.fit(X_train, y_train)


# In[8]:


# Step 13: Evaluating the model
from sklearn.metrics import accuracy_score, classification_report

# Predicting on test data
y_pred = model.predict(X_test)

# Evaluating accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))


# In[9]:


# Step 14: Building the Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

# Fit the model
rf_model.fit(X_train, y_train)

# Predicting on test data
rf_pred = rf_model.predict(X_test)

# Evaluating accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Model Accuracy: {rf_accuracy:.2f}")

# Classification report
print(classification_report(y_test, rf_pred))


# In[10]:


# Step 15: Hyperparameter Tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Display best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)


# In[11]:


# Step 16: Applying the best parameters to the Random Forest model
best_rf_model = RandomForestClassifier(max_depth=10, min_samples_split=2, n_estimators=50, random_state=42)

# Fit the optimized model
best_rf_model.fit(X_train, y_train)

# Predicting on test data
best_rf_pred = best_rf_model.predict(X_test)

# Evaluating the optimized model
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
print(f"Optimized Random Forest Model Accuracy: {best_rf_accuracy:.2f}")

# Classification report
print(classification_report(y_test, best_rf_pred))


# In[12]:


# Step 1: Calculate Feature Importance using the optimized Random Forest model
importances = best_rf_model.feature_importances_
features = X.columns

# Creating a DataFrame to display the importance
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
feature_importance_df


# In[13]:


# Step 2: Creating new features based on existing ones
# Adding a feature for hours per tenure (average monthly hours divided by tenure)
data['hours_per_tenure'] = data['average_monthly_hours'] / (data['tenure'] + 1)  # Adding 1 to avoid division by zero

# Check the first few rows to see the new feature
data.head()


# In[14]:


# Step 3: Updating the features set to include the new feature
X = data.drop('left', axis=1)  # Ensure 'left' is still the target variable

# Splitting the updated data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Re-training the optimized Random Forest model with the new feature
best_rf_model.fit(X_train, y_train)

# Predicting on the updated test data
new_rf_pred = best_rf_model.predict(X_test)

# Evaluating the updated model
new_rf_accuracy = accuracy_score(y_test, new_rf_pred)
print(f"Updated Random Forest Model Accuracy with New Feature: {new_rf_accuracy:.2f}")

# Classification report
print(classification_report(y_test, new_rf_pred))


# In[6]:


# Step 0: Load the dataset
import pandas as pd

# Load the data from the CSV file
file_path = r'C:\Users\TheBlessed\Documents\GADAC\gada_cap.csv'
data = pd.read_csv(file_path)

# Step 1: Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

# Features and target variable
X = data.drop('left', axis=1)  # Dropping the target column only
y = data['left']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Hyperparameter tuning for Decision Tree using Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define parameter grid for Decision Tree
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Initialize Grid Search with Decision Tree
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model with Grid Search
grid_search_dt.fit(X_train, y_train)

# Get the best parameters and accuracy
print("Best Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best Cross-Validation Accuracy:", grid_search_dt.best_score_)

# Train Decision Tree with best parameters
best_dt_model = grid_search_dt.best_estimator_
best_dt_model.fit(X_train, y_train)

# Predicting on test data
best_dt_pred = best_dt_model.predict(X_test)

# Evaluating the optimized Decision Tree model
best_dt_accuracy = accuracy_score(y_test, best_dt_pred)
print(f"Optimized Decision Tree Model Accuracy: {best_dt_accuracy:.2f}")

# Classification report
print(classification_report(y_test, best_dt_pred))


# In[7]:


# Step 0: Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data from the CSV file
file_path = r'C:\Users\TheBlessed\Documents\GADAC\gada_cap.csv'
data = pd.read_csv(file_path)

# Encode the categorical 'department' column
label_encoder = LabelEncoder()
data['department'] = label_encoder.fit_transform(data['department'])

# Features and target variable
X = data.drop('left', axis=1)  # Dropping the target column only
y = data['left']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define parameter grid for Decision Tree
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Initialize Grid Search with Decision Tree
grid_search_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model with Grid Search
grid_search_dt.fit(X_train, y_train)

# Get the best parameters and accuracy
print("Best Parameters for Decision Tree:", grid_search_dt.best_params_)
print("Best Cross-Validation Accuracy:", grid_search_dt.best_score_)

# Train Decision Tree with best parameters
best_dt_model = grid_search_dt.best_estimator_
best_dt_model.fit(X_train, y_train)

# Predicting on test data
best_dt_pred = best_dt_model.predict(X_test)

# Evaluating the optimized Decision Tree model
best_dt_accuracy = accuracy_score(y_test, best_dt_pred)
print(f"Optimized Decision Tree Model Accuracy: {best_dt_accuracy:.2f}")

# Classification report
print(classification_report(y_test, best_dt_pred))


# In[8]:


# Step: Training and tuning the Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Initialize Grid Search with Random Forest
grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model with Grid Search
grid_search_rf.fit(X_train, y_train)

# Get the best parameters and accuracy
print("Best Parameters for Random Forest:", grid_search_rf.best_params_)
print("Best Cross-Validation Accuracy:", grid_search_rf.best_score_)

# Train Random Forest with best parameters
best_rf_model = grid_search_rf.best_estimator_
best_rf_model.fit(X_train, y_train)

# Predicting on test data
best_rf_pred = best_rf_model.predict(X_test)

# Evaluating the optimized Random Forest model
best_rf_accuracy = accuracy_score(y_test, best_rf_pred)
print(f"Optimized Random Forest Model Accuracy: {best_rf_accuracy:.2f}")

# Classification report
print(classification_report(y_test, best_rf_pred))


# In[ ]:




