#!/usr/bin/env python
# coding: utf-8

# # **Capstone project: Providing data-driven suggestions for HR**

# ## Description and deliverables
# 
# This capstone project is an opportunity for you to analyze a dataset and build predictive models that can provide insights to the Human Resources (HR) department of a large consulting firm.
# 
# Upon completion, you will have two artifacts that you would be able to present to future employers. One is a brief one-page summary of this project that you would present to external stakeholders as the data professional in Salifort Motors. The other is a complete code notebook provided here. Please consider your prior course work and select one way to achieve this given project question. Either use a regression model or machine learning model to predict whether or not an employee will leave the company. The exemplar following this actiivty shows both approaches, but you only need to do one.
# 
# In your deliverables, you will include the model evaluation (and interpretation if applicable), a data visualization(s) of your choice that is directly related to the question you ask, ethical considerations, and the resources you used to troubleshoot and find answers or solutions.
# 

# # **PACE stages**
# 


# ## **Pace: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following:

# ### Understand the business scenario and problem
# 
# The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?
# 
# Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.
# 
# If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

# ### Familiarize yourself with the HR dataset
# 
# The dataset that you'll be using in this lab contains 15,000 rows and 10 columns for the variables listed below. 
# 
# **Note:** you don't need to download any data to complete this lab. For more information about the data, refer to its source on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv).
# 
# Variable  |Description |
# -----|-----|
# satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
# last_evaluation|Score of employee's last performance review [0&ndash;1]|
# number_project|Number of projects employee contributes to|
# average_monthly_hours|Average number of hours employee worked per month|
# time_spend_company|How long the employee has been with the company (years)
# Work_accident|Whether or not the employee experienced an accident while at work
# left|Whether or not the employee left the company
# promotion_last_5years|Whether or not the employee was promoted in the last 5 years
# Department|The employee's department
# salary|The employee's salary (U.S. dollars)

# 💭
# ### Reflect on these questions as you complete the plan stage.
# 
# *  Who are your stakeholders for this project?
# - What are you trying to solve or accomplish?
# - What are your initial observations when you explore the data?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 
# 

# [Double-click to enter your responses here.]

# ## Step 1. Imports
# 
# *   Import packages
# *   Load dataset
# 
# 

# ### Import packages

# In[42]:


# Import packages
### YOUR CODE HERE ### 
# For data manipulation
import numpy as np
import pandas as pd

# For data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree
from sklearn import metrics

# For saving models
import pickle


# ### Load dataset
# 
# `Pandas` is used to read a dataset called **`HR_capstone_dataset.csv`.**  As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[2]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

# Load dataset into a dataframe
### YOUR CODE HERE ###
df0 = pd.read_csv("HR_capstone_dataset.csv")


# Display first few rows of the dataframe
### YOUR CODE HERE ###
df0.head(10)


# ## Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# - Understand your variables
# - Clean your dataset (missing data, redundant data, outliers)
# 
# 

# ### Gather basic information about the data

# In[3]:


# Gather basic information about the data
### YOUR CODE HERE ###
df0.dtypes


# ### Gather descriptive statistics about the data

# In[5]:


# Gather descriptive statistics about the data
### YOUR CODE HERE ###
df0.describe()


# ### Rename columns

# As a data cleaning step, rename the columns as needed. Standardize the column names so that they are all in `snake_case`, correct any column names that are misspelled, and make column names more concise as needed.

# In[4]:


# Display all column names
### YOUR CODE HERE ###
list(df0.columns.values)


# In[5]:


# Rename columns as needed
### YOUR CODE HERE ###
df0 = df0.rename(columns={'satisfaction_level': 'satisfaction','last_evaluation':'evaluation','average_montly_hours':'month_hours', 'time_spend_company':'tenure', 'Work_accident':'accident', 'promotion_last_5years':'promotion','Department':'department' })

# Display all column names after the update
### YOUR CODE HERE ###
list(df0.columns.values)


# ### Check missing values

# Check for any missing values in the data.

# In[6]:


# Check for missing values
### YOUR CODE HERE ###
df0.isna().sum()
df0.info()


# ### Check duplicates

# Check for any duplicate entries in the data.

# In[7]:


# Check for duplicates
### YOUR CODE HERE ###
duplicates = df0[df0.duplicated()==True]


# In[8]:


# Inspect some rows containing duplicates as needed
### YOUR CODE HERE ##
duplicates.head(10)


# In[9]:


# Drop duplicates and save resulting dataframe in a new variable as needed
### YOUR CODE HERE ###
df = df0.drop_duplicates()

# Display first few rows of new dataframe as needed
### YOUR CODE HERE ###
df.head(10)


# ### Check outliers

# Check for outliers in the data.

# In[10]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
### YOUR CODE HERE ###
sns.boxplot(x=df['tenure'])


# In[21]:


# Determine the number of rows containing outliers
### YOUR CODE HERE ###
df['tenure'].describe()
df[df['tenure']>5].value_counts()


# Certain types of models are more sensitive to outliers than others. When you get to the stage of building your model, consider whether to remove outliers, based on the type of model you decide to use.

# # pAce: Analyze Stage
# - Perform EDA (analyze relationships between variables)
# 
# 

# 💭
# ### Reflect on these questions as you complete the analyze stage.
# 
# - What did you observe about the relationships between variables?
# - What do you observe about the distributions in the data?
# - What transformations did you make with your data? Why did you chose to make those decisions?
# - What are some purposes of EDA before constructing a predictive model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 
# 

# [Double-click to enter your responses here.]

# ## Step 2. Data Exploration (Continue EDA)
# 
# Begin by understanding how many employees left and what percentage of all employees this figure represents.

# In[23]:


# Get numbers of people who left vs. stayed
### YOUR CODE HERE ###
df['left'].value_counts()
# Get percentages of people who left vs. stayed
### YOUR CODE HERE ###
df['left'].value_counts(normalize = True)


# ### Data visualizations

# Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

# In[25]:


# Create a plot as needed
### YOUR CODE HERE ###
sns.histplot(data = df, x='month_hours', hue='left')


# In[26]:


# Create a plot as needed
### YOUR CODE HERE ###
sns.histplot(data = df, x='number_project', hue='left')


# In[27]:


# Create a plot as needed
### YOUR CODE HERE ###
sns.histplot(data = df, x='evaluation', hue='left')


# In[30]:


# Create a plot as needed
### YOUR CODE HERE ###
sns.histplot(data = df, x='tenure', hue='left', discrete = 1)


# In[11]:


# Create a plot as needed
### YOUR CODE HERE ###
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);


# In[ ]:


# Create a plot as needed
### YOUR CODE HERE ###


# In[ ]:


# Create a plot as needed
### YOUR CODE HERE ###


# In[ ]:


# Create a plot as needed
### YOUR CODE HERE ###


# ### Insights

# [What insights can you gather from the plots you created to visualize the data? Double-click to enter your responses here.]

# # paCe: Construct Stage
# - Determine which models are most appropriate
# - Construct the model
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# 

# 🔎
# ## Recall model assumptions
# 
# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the constructing stage.
# 
# - Do you notice anything odd?
# - Which independent variables did you choose for the model and why?
# - Are each of the assumptions met?
# - How well does your model fit the data?
# - Can you improve it? Is there anything you would change about the model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# [Double-click to enter your responses here.]

# ## Step 3. Model Building, Step 4. Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.

# [Double-click to enter your responses here.]

# ### Identify the types of models most appropriate for this task.

# [Double-click to enter your responses here.]

# ### Modeling
# 
# Add as many cells as you need to conduct the modeling process.

# In[32]:


### YOUR CODE HERE ###
###Handle catorgorical variables
df['salary'] = df['salary'].astype('category').cat.set_categories(['low', 'medium', 'high']).cat.codes
df = pd.get_dummies(df, drop_first = False)
df.head()

## create y and X
y= df['left']
y.head()

X = df.drop('left', axis =1)
X.head()


# In[33]:


# Perform the split operation on your data.
# Assign the outputs as follows: X_train, X_test, y_train, y_test.

### YOUR CODE HERE ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state = 0)


# In[34]:


# Define xgb to be your XGBClassifier.

### YOUR CODE HERE ###
xgb = XGBClassifier(objective = 'binary:logistic', random_state = 0)


# In[35]:


# Define parameters for tuning as `cv_params`.

### YOUR CODE HERE ###
cv_params = {'max_depth': [3,5, None],
              'min_child_weight': [3, 5],
              'learning_rate': [0.1, 0.2, 0.3],
              'n_estimators': [300, 500],
              'subsample': [0.7],
              'colsample_bytree': [0.7]
              }


# In[36]:


# Define your criteria as `scoring`.

### YOUR CODE HERE ###
scoring = {'accuracy','precision', 'recall','f1'}


# In[37]:


# Construct your GridSearch.

### YOUR CODE HERE ###
xgb_cv = GridSearchCV(xgb,
                      cv_params,
                      scoring = scoring,
                      cv = 5,
                      refit = 'f1'
                     )


# In[38]:


get_ipython().run_cell_magic('time', '', '# fit the GridSearch model to training data\n\n### YOUR CODE HERE ###\nxgb_cv = xgb_cv.fit(X_train, y_train)\nxgb_cv')


# In[39]:


# Use `pickle` to save the trained model.

### YOUR CODE HERE ###
pickle.dump('xgb_cv', open('xgb_cv.sav', 'wb'))


# In[40]:


# Apply your model to predict on your test data. Call this output "y_pred".

### YOUR CODE HERE ###
y_pred = xgb_cv.predict(X_test)


# In[43]:


# 1. Print your accuracy score.

### YOUR CODE HERE ###
ac_score = metrics.accuracy_score(y_test, y_pred)
print('accuracy score:', ac_score)

# 2. Print your precision score.

### YOUR CODE HERE ###
pc_score = metrics.precision_score(y_test, y_pred)
print('precision score:', pc_score)

# 3. Print your recall score.

### YOUR CODE HERE ###
rc_score = metrics.recall_score(y_test, y_pred)
print('recall score:', rc_score)

# 4. Print your f1 score.

### YOUR CODE HERE ###
f1_score = metrics.f1_score(y_test, y_pred)
print('f1 score:', f1_score)


# In[44]:


# Construct and display your confusion matrix.

# Construct the confusion matrix for your predicted and test values.

### YOUR CODE HERE ###

cm = metrics.confusion_matrix(y_test, y_pred)

# Create the display for your confusion matrix.

### YOUR CODE HERE ###

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb_cv.classes_)

# Plot the visual in-line.

### YOUR CODE HERE ###

disp.plot()


# In[45]:


# Plot the relative feature importance of the predictor variables in your model.

### YOUR CODE HERE ###

plot_importance(xgb_cv.best_estimator_)


# # pacE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders
# 
# 

# ✏
# ## Recall evaluation metrics
# 
# - **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
# - **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
# - **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
# - **Accuracy** measures the proportion of data points that are correctly classified.
# - **F1-score** is an aggregation of precision and recall.
# 
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the executing stage.
# 
# - What key insights emerged from your model(s)?
# - What business recommendations do you propose based on the models built?
# - What potential recommendations would you make to your manager/company?
# - Do you think your model could be improved? Why or why not? How?
# - Given what you know about the data and the models you were using, what other questions could you address for the team?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# Double-click to enter your responses here.

# ## Step 4. Results and Evaluation
# - Interpret model
# - Evaluate model performance using metrics
# - Prepare results, visualizations, and actionable steps to share with stakeholders
# 
# 
# 

# ### Summary of model results
# 
# [Double-click to enter your summary here.]

# ### Conclusion, Recommendations, Next Steps
# 
# [Double-click to enter your conclusion, recommendations, and next steps here.]

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
