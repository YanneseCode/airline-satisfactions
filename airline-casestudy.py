# %% [markdown]
# # **Capstone Project: Providing Data-driven Suggestions to the Invistico Airline to improve their customer satisfaction**

# # Pace: Plan Stage
# 
# - Understand your data in the problem context
# - Consider how your data will best address the business need
# - Contextualize & understand the data and the problem
# 

# %% [markdown]
# 🗒
# ### Understand the business scenario and problem
# 
# The dataset is derived from Kaggle.com. In this [dataset](https://www.kaggle.com/datasets/mohaimenalrashid/invistico-airline). It consists of survey results and demographic/personal information related to 129,880 customers. Originally, there are 22 variables, the dataset is later updated to 23 variables (This case study, it has only consists of 22 columns without "gender" column). Within this dataset, there are about 4 variables are categorical data, whereas the rest of 17 columns were 17 variables. To begin with this dataset, we would need to carry out the explanatory data analysis (EDA), in a way to access the demographic condition of the dataset. Then, the next step would be performing cleaning process in the data to make sure there is no missing duplicated or noises that would affect our proceeding analyse process. 
# 
# When proceed to the analyze stage, *Regression* and *Classification* algorithmn would be applied in the Machine learning model to understand their customers attribute and behaviour on their satisfaction level towards the Invistico Airline. 
# In this case, The label (Dependent variable) would be the first column of “satisfaction” , whereas the rest columns would be considered as the features (Independent variables) in this project study. As diving into the EDA analysis in this dataset, the satisfaction proportion of the dataset was likely equally distributed,which about 55% was “satisfied” customers, and 45% “dissatisfied” customers were captured. Hence, the dataset is presumably balance enough to avoid overfitting/ underfitting outcomes when running for the model training and testing processes. Still, this project will end it with tuning process to avoid such misleadind practices.

# %% [markdown]
# ### Understand the Invisco Airline datasets
# 
# There are 129,880 rows (Exclude header row), 22 columns, and these variables: 
# 
# Variable  |Data Type |Description |
# -----|-----|-----| 
# satisfaction|Categorical|Customers' satisfaction level towards the airline services, either "satisfied" or "dissatisfied" [0&ndash;1]|
# Customer Type|Categorical|There are 2 customer type, either "disloyal" or "loyal" customer [0&ndash;1]|
# Age|Numerical|The age range from 7 to 85 years old|
# Type of Travel|Categorical|There are 2 travel type, "Personal" or "Business" Travel [0&ndash;1]|
# Class|Categorical|The Airline class has taken by passengers, includes "Business", "Eco" and "Eco-Plus" [0&ndash;2]|
# Flight Distance|Numerical|Flight distance for passengers ranging from 50 to 6951 mileage distance|
# Seat Comfort|Numerical|passengers' ratings: [0&ndash;5]|
# Departure/Arrival time convenient|Numerical|passengers' ratings: [0&ndash;5]|
# Food and drink|Numerical|passengers' ratings: [0&ndash;5]|
# Gate location|Numerical|passengers' ratings: [0&ndash;5]|
# Inflight wifi service|Numerical|passengers' ratings: [0&ndash;5]|
# Inflight entertainment|Numerical|passengers' ratings: [0&ndash;5]|
# Online support|Numerical|assengers' ratings: [0&ndash;5]|
# Ease of Online booking|Numerical|passengers' ratings: [0&ndash;5]|
# On-board service|Numerical|passengers' ratings: [0&ndash;5]|
# Leg room service|Numerical|passengers' ratings: [0&ndash;5]|
# Baggage handling|Numerical|passengers' ratings: [0&ndash;5]|
# Checkin service|Numerical|passengers' ratings: [0&ndash;5]|
# Cleanliness|Numerical|passengers' ratings: [0&ndash;5]|
# Online boarding|Numerical|passengers' ratings: [0&ndash;5]|
# Departure Delay in Minutes|Numerical|Departure delay time frame measured in minutes, the longest time is 1591 minutes|
# Arrival Delay in Minutes|Numerical|Arrival delay time frame measured in minutes, the longest time is 1584 minutes|

# %% [markdown]
# ## Step 1. Imports
# 
# *   Import packages
# *   Load dataset
# 
# 

# %% [markdown]
# ### Import packages

# %%
# Import packages

# For data manipulation
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# For data visualization
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

# For data modeling for supervised-learning algorithmn
# !pip install xgboost
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

# For saving models
import pickle

# %% [markdown]
# ### Load dataset

# %%
# Load dataset into a dataframe
df = pd.read_csv("Invistico_Airline.csv")

# Display first few rows of the dataframe
df.head()

# %% [markdown]
# ## Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# - Understand the data variables
# - Clean your dataset (missing data, redundant data, outliers) 
# 
# 

# %% [markdown]
# ### Gather basic information about the data

# %%
# Gather basic information about the data
df.info()

# %%
# Gather descriptive statistics about the data
df.describe()

# %% [markdown]
# As a data cleaning step, rename the columns as needed. Standardize the column names so that to eliminate those header names with empty spaces and capital letter case. I have replaced the empty space with underscore, and also small caps all header names.

# %%
# Display all column names
df.columns

# %%
# Rename columns as needed
df0 = df.rename(columns={'Customer Type': 'customer_type',
                          'Age': 'age', 'Type of Travel': 'travel_type', 'Class': 'class','Flight Distance': 'flight_distance',
                          'Seat comfort': 'seat_comfort', 'Departure/Arrival time convenient': 'dept_arr_time_convenient', 
                          'Food and drink':'food_service', 'Gate location': 'gate_location', 'Inflight wifi service': 'inflight_wifi',
                          'Inflight entertainment': 'inflight_entertainment', 'Online support':'online_support', 'Ease of Online booking':'ease_online_booking',
                          'On-board service': 'onboard', 'Leg room service':'legroom', 'Baggage handling': 'baggage_handling',
                          'Checkin service':'checkin', 'Cleanliness': 'cleanliness', 'Online boarding':'online_boarding',
                          'Departure Delay in Minutes': 'departure_delaytime', 'Arrival Delay in Minutes': 'arrival_delaytime'})

# Display all column names after the update
df0.columns

# %%
# Check for missing values
df0.isna().sum()

# %% [markdown]
# There are 393 missing values on the last column named "arrival_delaytime" derived from the dataset. 

# %%
# Remove rows where a specific column has missing values
df0.dropna(subset=['arrival_delaytime'], inplace=True)

# %%
# Check again for the missing values for all columns
df0.isna().sum()

# %%
# Check for duplicates
# method 1: Use sum method
df0.duplicated().sum()

# %%
# method 2
# Inspect some rows containing duplicates as needed in table format
df0[df0.duplicated()].head()

# %%
# Create a boxplot to visualize distribution of `flight_distance` and detect any outliers
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for flight distance', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df0['flight_distance'])
plt.show()

# %% [markdown]
# The boxplot above shows that there are quite a lot of outliers in the `flight_distance` variable. 
# 
# It would be helpful to investigate how many rows in the data contain outliers in the `flight_distance` column.

# %%
# Determine the number of rows containing outliers 

# Compute the 25th percentile value in `flight_distance`
percentile25 = df0['flight_distance'].quantile(0.25)

# Compute the 75th percentile value in `flight_distance`
percentile75 = df0['flight_distance'].quantile(0.75)

# Compute the interquartile range in `flight_distance`
iqr = percentile75 - percentile25

# Define the upper limit and lower limit for non-outlier values in `flight_distance`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in `flight_distance`
outliers = df0[(df0['flight_distance'] > upper_limit) | (df0['flight_distance'] < lower_limit)]

# Count how many rows in the data contain outliers in `flight_distance`
print("Number of rows in the data containing outliers in `flight_distance`:", len(outliers))

# %% [markdown]
# Based on the observations, there are about 2,575 outliers has been captured. Whether to remove those outliers, it is depending on the models that are applied are sensitive to those outliers.In this current EDA stage, I will hold still to this data till the stage that building our model.

# %%
# plot a bar plot for each ratings feature count
rating_features = ['seat_comfort', 'dept_arr_time_convenient', 'food_service', 'gate_location','inflight_wifi','inflight_entertainment','online_support','ease_online_booking', 'onboard','legroom','baggage_handling','checkin','cleanliness','online_boarding']

for col in rating_features:
    counts = df0[col].value_counts().sort_index()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca()
    counts.plot.bar(ax = ax, color='steelblue')
    ax.set_title(col + ' counts')
    ax.set_xlabel(col) 
    ax.set_ylabel("Frequency")
plt.show()

# %%
# Dummy encode categoricals
satisfy_df0 = pd.get_dummies(df0, drop_first='True')
satisfy_df0.head()

# %%
satisfy_df0.describe()

# %%
# show the encoded columns again
satisfy_df0.columns 

# %%
satisfy_df1 = satisfy_df0.rename(columns={'satisfaction_satisfied': 'satisfied', 'customer_type_disloyal Customer':'disloyal_customer' ,
       'travel_type_Personal Travel': 'personal_travel' , 'class_Eco':'class_eco', 'class_Eco Plus': 'class_ecoplus'})

satisfy_df1.columns

# %% [markdown]
# ### Split the dataset 

# %%
# Define the y (target) variable
y = satisfy_df1["satisfied"]

# Define the X (predictor) variables
X = satisfy_df1.copy()
X = X.drop("satisfied", axis = 1)

# Create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Create train & validate data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)

# # pAce: Analyze Stage
# - Perform EDA (analyze relationships between variables) 

# %%
# Create a plot as needed 
# Duplicate a copy file df2 for visualization
df2 = satisfy_df1.copy()

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df2, x='flight_distance', y='satisfied', hue='disloyal_customer', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Flight distance by customer loyalty', fontsize='14')

# Create histogram showing distribution of `satisfaction`, comparing customers who satisfy to the airline services
not_satisfy = df2[df2['satisfied']==0]['flight_distance'] 
satisfy = df2[df2['satisfied']==1]['flight_distance']
sns.histplot(data=df2, x='flight_distance', hue='satisfied', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of flight distance histogram', fontsize='14')

# Display the plots
plt.show()

# %% [markdown]
# Based on the observation above, most of the travelers, regardless of their loyalty status, are within the same range of flight mileage, which fall between 1000 and 3000km with mean distance of 2000km. 
# 
# 1. There are two groups of customers who loyal and disloyal to the airline company: (A) those who customers who accumulated the flying mileage roughly ranging from 1200km until 2500km with airline services, were mostly disatisfied with the services, and (B) those who who accumulated the flying mileage outside the range of group A (< 1200, and >= 2500km), were generally satisfied with the services. It is possible to believe that the quality of the services were influenced by the customers frequency of in certain travel area and distance. There is a need for further investigation whether which countries that most customers that traveling within the group A that constitute to such higher disatisfactory level.
# 
# 2. Both disloyal and loyal customer with mean flight distance of 2000km projects shared almost the equal proportion of satisfaction level. Within the group of satisfied customers group, there were about half size of them were disloyal customes. Therefore, the company should contribute more effort in converting those customers to their loyalty customers in building up more sales. On the other side of it, there were about more than the half size of loyalty customers have express their disatisfaction towards their services, the company should paying more attention to retent those customers in avoiding tremendous loss coming their way.
# 
# In concluded from the above graphs, we learned that travellers with the flight distance of 1200-2500km were leaving more unsatisfactory review to the airline company.

# %%
# Get value counts of customer satisfactions to overall services
df2[(df2['flight_distance']>=1200) & (df2['flight_distance']<=2500) ].groupby('satisfied').size().reset_index(name='counts')

# %%
# Calculate mean and median satisfaction scores of employees who left and those who stayed
df2.groupby(['disloyal_customer'])['satisfied'].agg([np.mean,np.median])

# %% [markdown]
# As expected, the mean satisfaction score of disloyal customer are lower than those of loyal customer. Interestingly, for those loyal customers,the mean satisfaction score appears to be slightly above the standard score of 0.5/1.0. This indicates that satisfaction levels among those who stayed might be skewed to the left. 
# 
# Next, we could examine customer types and classes proportions by using bar chart.

# %%
# Set figure and axes
fig, ax = plt.subplots(figsize=(22, 8))

# Plot traveller's type histogram by all age
sns.histplot(data=df0, x='age', hue='travel_type', discrete=1,
             hue_order=['Business travel', 'Personal Travel'], multiple='dodge', shrink=.5, ax=ax)
ax.set_title('Travellers Type histogram by Age', fontsize='14')


# %% [markdown]
# Based on the histogram result above, it is shockingly revealed that the numbers of personal travellers are in consistent pattern spreading across all ages, with about 600-700 of people. Aside from that, most of the business travellers were ranging across the age from 20+ up to 60 years old. Further, there were no personal travellers were captured from the age of 70 onwards.

# %%
# Set figure and axes
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (22,8))

# Define short-tenured employees
below_30 = df0[df0['age']< 30]

# Define long-tenured employees
above_30 = df0[df0['age'] >= 30]

# Plot short-tenured histogram
sns.histplot(data=below_30, x='age', hue='travel_type', discrete=1, 
             hue_order=['Business travel', 'Personal Travel'], multiple='dodge', shrink=.5, ax=ax1)
ax1.set_title('Travellers Type histogram by Age (below 30)', fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=above_30, x='age', hue='travel_type', discrete=1, 
             hue_order=['Business travel', 'Personal Travel'], multiple='dodge', shrink=.5, ax=ax2)
ax2.set_title('Travellers Type histogram by Age (above 30)', fontsize='14');

# %%
# Display counts for customer satisfactions
df2["satisfied"].value_counts()

# %%
# Display counts for customer loyalty
df2["disloyal_customer"].value_counts()

# %%
# Set figure and axes
fig, (ax1, ax2) = plt.subplots(1,2,figsize = (22,8))

# set tick labels
ticks = [0, 1, 2, 3]
#ax1.set_xticks(ticks)
#ax1.set_xticklabels(['First', 'Second', 'Third', 'Fourth'], rotation='horizontal')

# Create histogram graph for number of customers' satisfactions by class
sns.histplot(data=df0, x='class', hue='satisfaction', discrete=1, 
             hue_order=['satisfied','dissatisfied'], multiple='dodge', shrink=.5, ax=ax1)
ax1.set_xticks(ticks, rotation='horizontal')
ax1.set_title('Counts of satisfaction by flight class', fontsize=14);

# Create histogram graph for number of customers' loyalty by class
sns.histplot(data=df0, x='class', hue='customer_type', discrete=1, 
             hue_order=['Loyal Customer','disloyal Customer'], multiple='dodge', shrink=.5, ax=ax2)
ax2.set_xticks(ticks, rotation='horizontal')
ax2.set_title('Counts of customer loyalty by flight class', fontsize=14);

# %%
# Create a plot as needed 

# Plot a correlation heatmap
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df0.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);

# %% [markdown]
# The correlation heatmap confirms that most of the features are popsitively correlated to each other, and whereas flight distance is negatively correlated with age. 

# # paCe: Construct Stage
# - Determine which models are most appropriate
# - Construct the model 
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data

# %% [markdown]
# ### Modeling Approach A: Logistic Regression Model
# 
# This approach covers implementation of Logistic Regression.

# %% [markdown]
# #### Logistic regression
# Note that binomial logistic regression suits the task because it involves binary classification.

# %% [markdown]
# Start by one-hot encoding the categorical variables as needed.

# %%
# One-hot encode the categorical variables as needed and save resulting dataframe in a new variable
df_enc = pd.get_dummies(df0, prefix=['satisfaction', 'customer_type', 'travel_type','class'], columns = ['satisfaction', 'customer_type', 'travel_type','class'], drop_first=False)

# Display the new dataframe
df_enc.head()

# %% [markdown]
# Create a heatmap to visualize how correlated variables are. Consider which variables you're interested in examining correlations between.

# %%
# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(20, 8))
sns.heatmap(df_enc[['age', 'flight_distance', 'seat_comfort', 'dept_arr_time_convenient',
       'food_service', 'gate_location', 'inflight_wifi',
       'inflight_entertainment', 'online_support', 'ease_online_booking',
       'onboard', 'legroom', 'baggage_handling', 'checkin', 'cleanliness',
       'online_boarding', 'departure_delaytime', 'arrival_delaytime','satisfaction_satisfied']].corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()

# %% [markdown]
# Create a stacked bar plot to visualize number of employees across department, comparing those who left with those who didn't.

# %%
# Create a stacked bart plot to visualize number of employees across department, comparing those who left with those who didn't
# In the legend, 0 (purple color) represents employees who did not leave, 1 (red color) represents employees who left
pd.crosstab(df0["class"], df0["customer_type"]).plot(kind ='bar',color='mr')
plt.title('Counts of customers who is loyal versus disloyal across classes')
plt.ylabel('Customer count')
plt.xlabel('Classes')
plt.show()

# %%
df0.head()

# %% [markdown]
# Since logistic regression is quite sensitive to outliers, it would be a good idea at this stage to remove the outliers in the `tenure` column that were identified earlier.

# %%
# Select rows without outliers in `tenure` and save resulting dataframe in a new variable
df_logreg = df_enc[(df_enc['flight_distance'] >= lower_limit) & (df_enc['flight_distance'] <= upper_limit)]

# Display first few rows of new dataframe
df_logreg.describe()

# %% [markdown]
# Isolate the outcome variable, which is the variable you want your model to predict.

# %%
# Isolate the outcome variable
y = df_logreg['satisfaction_satisfied']

# Display first few rows of the outcome variable
y.head() 

# %%
# Select the features you want to use in your model
X = df_logreg[['age', 'flight_distance', 'seat_comfort', 'dept_arr_time_convenient',
       'food_service', 'gate_location', 'inflight_wifi',
       'inflight_entertainment', 'online_support', 'ease_online_booking',
       'onboard', 'legroom', 'baggage_handling', 'checkin', 'cleanliness',
       'online_boarding', 'departure_delaytime', 'arrival_delaytime',]]

# Display the first few rows of the selected features 
X.head()

# %%
x_predictors = [ 'seat_comfort', 'dept_arr_time_convenient',
       'food_service', 'gate_location', 'inflight_wifi',
       'inflight_entertainment', 'online_support', 'ease_online_booking',
       'onboard', 'legroom', 'baggage_handling', 'checkin', 'cleanliness',
       'online_boarding']

# %%
def create_pivot_graphs(df0, x_column):
    _df_plot = df0.groupby([x_column, 'satisfaction']).size() \
    .reset_index().pivot(columns='satisfaction', index=x_column, values=0)
    return _df_plot


# %%
fig, ax = plt.subplots(4,2 , figsize=(20,30))
axe = ax.ravel()

for i in range(0,14):
    create_pivot_graphs(df0, x_predictors[i]).plot(kind='bar',stacked=True, ax=axe[i],color=["#F6CF71","#66C5CC"])
    plt.xlabel(x_predictors[i])
    axe[i].set_ylabel('Count of Flight travellers satisfaction')
fig.show()

# %% [markdown]
# ## Training and Testing the dataset

# %%
# make another copy to label the dataset
df_air = df0.copy()
df_air.head()

# %%
def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['column','dtypes','nunique','sum_null']
    return df_check 
check(df_air)

# %%
# Encoding the features
for i in df_air.select_dtypes(include=['object']):
        le = LabelEncoder()
        df_air[i] = le.fit_transform(df_air[i])

df_air.head()

# %%
df_air.isnull().sum()

# %%
# Find out the Outliers
Q1 = df_air.quantile(0.25)
Q3 = df_air.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

# %%
fig = plt.figure(figsize=(12,18))
for i in range(len(df_air.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=df_air.iloc[:,i])

plt.tight_layout()
plt.show()

# %%
from collections import Counter
def detect_outliers(df_air,features):
    outlier_indices=[]
    
    for c in features:
        Q1=np.percentile(df_air[c],25)
        
        Q3=np.percentile(df_air[c],75)
        
        IQR= Q3-Q1
        
        outlier_step= IQR * 1.5
        
        outlier_list_col = df_air[(df_air[c]< Q1 - outlier_step)|( df_air[c] > Q3 + outlier_step)].index
        
        outlier_indices.extend(outlier_list_col)
    
    outliers_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i , v in outliers_indices.items() if v>2 )
    return multiple_outliers

# %%
df_air.columns

# %%
if "age" in df_air.columns:
    df_air.loc[detect_outliers(df_air, ['age', 'flight_distance', 'seat_comfort', 'dept_arr_time_convenient',
                                        'food_service', 'gate_location', 'inflight_wifi',
                                        'inflight_entertainment', 'online_support', 'ease_online_booking',
                                        'onboard', 'legroom', 'baggage_handling', 'checkin', 'cleanliness',
                                        'online_boarding', 'departure_delaytime', 'arrival_delaytime'])]
else:
    print("The column 'age' does not exist in the dataframe.")


# %%
df_air = df_air.drop(detect_outliers(df_air,[ 'age', 'flight_distance', 'seat_comfort', 'dept_arr_time_convenient',
                                        'food_service', 'gate_location', 'inflight_wifi',
                                        'inflight_entertainment', 'online_support', 'ease_online_booking',
                                        'onboard', 'legroom', 'baggage_handling', 'checkin', 'cleanliness',
                                        'online_boarding', 'departure_delaytime', 'arrival_delaytime']),axis = 0).reset_index(drop = True)

# %%
fig = plt.figure(figsize=(12,18))
for i in range(len(df_air.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=df_air.iloc[:,i],color="#FFDDDD")

plt.tight_layout()
plt.show()

# %%
df_air.shape

# %%
# Normalization of the data
from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
r_scaler.fit(df_air)
modified_data = pd.DataFrame(r_scaler.transform(df_air), columns=df_air.columns)
modified_data.head()

# %% [markdown]
# Split the data into training set and testing set.

# %%
# Split the data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ✏
# ## Definition of Evaluation metrics
# 
# - **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example. 
# - **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
# - **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
# - **Accuracy** measures the proportion of data points that are correctly classified.
# - **F1-score** is an aggregation of precision and recall.
# 

# %% [markdown]
# ## Construct a Logistic regression model

# %%
# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)

# %% [markdown]
# Test the logistic regression model: use the model to make predictions on the test set.

# %%
# Use the logistic regression model to get predictions on the test set
y_pred = log_clf.predict(X_test)

# %% [markdown]
# Create a confusion matrix to visualize the results of the logistic regression model. 

# %%
# Compute values for confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.show()


# %% [markdown]
# Create a classification report that includes precision, recall, f1-score, and accuracy metrics to evaluate the performance of the logistic regression model.

# %% [markdown]
# Check the class balance in the data. In other words, check the value counts in the `satisfaction_satisfied` column. Since this is a binary classification task, the class balance informs the way you interpret accuracy metrics.

# %%
df_logreg['satisfaction_satisfied'].value_counts(normalize=True)

# %% [markdown]
# There is an approximately 54%-45% split. So the data is considerably balanced. If it was more severely imbalanced, we might want to resample the data to make it more balanced. In this case, you can use this data without modifying the class balance and continue evaluating the model.

# %%
# Create classification report for logistic regression model
target_names = ['Predicted to be satisfied', 'Predicted to be dissatisfied']
print(classification_report(y_test, y_pred, target_names=target_names))

# %% [markdown]
# The classification report above shows that the logistic regression model achieved a precision of 78%, recall of 77%, f1-score of 77% (all weighted averages), and accuracy of 77%.

# %% [markdown]
# ### Modeling Approach B: Tree-based Model
# This approach covers implementation of Decision Tree and Random Forest. 

# %% [markdown]
# Split the data into training, validating, and testing sets.

# %%
# Create test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# Create train & validate data
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=0)

# %% [markdown]
# #### Decision tree - Round 1

# %% [markdown]
# Construct a decision tree model and set up cross-validated grid-search to exhuastively search for the best model parameters.

# %%
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}


# %%
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Instantiate GridSearch
tree_cv = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# %% [markdown]
# Fit the decision tree model to the training data.

# %%
%%time
tree_cv.fit(X_tr, y_tr)

# %% [markdown]
# Identify the optimal values for the decision tree parameters.

# %%
# Check best parameters
tree_cv.best_params_

# %% [markdown]
# Identify the best AUC score achieved by the decision tree model on the training set.

# %%
# Check best AUC score on CV
tree_cv.best_score_

# %% [markdown]
# The results of 96.58% is a strong AUC score , which shows that this model can predict customer satisfactions who will satisfy the airline services.
# 
# Next, we can write a function that will help you extract all the scores from the grid search. 

# %% [markdown]
# ### Modeling approach C- XGBoost Classifier Model 

# %%
# Instantiate model
xgb = XGBClassifier(objective='binary:logistic', random_state=0)   

# Instantiate GridSearch
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring,cv=4, refit='roc_auc')

# %%
%%time
xgb_cv.fit(X_tr, y_tr)

# %%
# Check best parameters
xgb_cv.best_params_

# %% [markdown]
# ### Logistic Regression - Model Scoring

# %%
# Instantiate GridSearch
log_cv = GridSearchCV(log_clf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# %%
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc
  
    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.  
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                 'precision': 'mean_test_precision',
                 'recall': 'mean_test_recall',
                 'f1': 'mean_test_f1',
                 'accuracy': 'mean_test_accuracy',
                 }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
  
    # Create table of results
    table = pd.DataFrame()
    table = table.append({'Model': model_name,
                        'AUC': auc,
                        'Precision': precision,
                        'Recall': recall,
                        'F1': f1,
                        'Accuracy': accuracy,
                        },
                        ignore_index=True
                       )
  
    return table

# %% [markdown]
# Use the function just defined to get all the scores from grid search.

# %%
# Get decision tree model CV scores
tree_cv_results = make_results('decision tree cv', tree_cv, 'auc')
tree_cv_results

# %%
# Get decision tree model CV scores
xgb_cv_results = make_results('XGBoost cv', xgb_cv, 'auc')
xgb_cv_results

# %%
# Collect validation scores 
tree_val_results = get_scores('Decision Tree val', tree_cv, X_val, y_val)
xgb_val_results = get_scores('XGBoost val', xgb_cv, X_val, y_val)
#log_val_results = get_scores('Logistic Regression val', log_cv, X_val, y_val)

# Concatenate validation scores into table
all_val_results = [tree_val_results, xgb_val_results]
all_val_results = pd.concat(all_val_results).sort_values(by='AUC', ascending=False)
all_val_results

# %% [markdown]
# All of these scores from the decision tree model are strong indicators of good model performance. 
# 
# Recall that decision trees can be vulnerable to overfitting, and random forests avoid overfitting by incorporating multiple trees to make predictions. You could construct a random forest model next.




# %% [markdown]
# #### Feature Engineering

# %% [markdown]
# #### Decision tree - Round 2

# %%
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)

# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }

# Assign a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

# %%
%%time
tree2.fit(X_tr, y_tr)

# %%
# Check best params
tree2.best_params_

# %%
# Check best AUC score on CV
tree2.best_score_

# %% [markdown]
# This model performs very well, even without satisfaction levels and detailed hours worked data. 
# 
# Next, check the other scores.

# %%
# Get all CV scores
tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
tree2_cv_results

# %% [markdown]
# Some of the other scores fell. That's to be expected given fewer features were taken into account in this round of the model. Still, the scores are very good.

# %% [markdown]
# #### Decision tree splits

# %%
# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree_cv.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()

# %% [markdown]
# #### Decision tree feature importance
# 
# You can also get feature importance from decision trees (see the [DecisionTreeClassifier scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_) for details).

# %%
#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree_importances = pd.DataFrame(tree_cv.best_estimator_.feature_importances_, columns=['gini_importance'], index=X.columns)
tree_importances = tree_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree_importances = tree_importances[tree_importances['gini_importance'] != 0]
tree_importances

# %% [markdown]
# You can then create a barplot to visualize the decision tree feature importances.

# %%
sns.barplot(data=tree_importances, x="gini_importance", y=tree_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Traveler's satisfactions", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()

# # pacE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders
# 
# 

# %% [markdown]
# ### Summary of model results
# 
# **Logistic Regression**
# 
# The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.
# 
# **Tree-based Machine Learning**
# 
# After conducting feature engineering, the decision tree model achieved AUC of 96.6%, precision of 92.23%, recall of 89.47.5%, f1-score of 90.83%, and accuracy of 90.15%, on the test set. The random forest modelslightly outperformed the decision tree model. 
# 
# **XGBoost Classifier Model**
# 
# After conducting feature engineering, the XGBoost classified model achieved AUC of 94.7%, precision of 95.9%, recall of 94.3%, f1-score of 95.1%, and accuracy of 94.7%, on the test set. The random forest modelslightly outperformed the decision tree model. 
# 
# **Conclusion**
# 
# In overall view of this airline company, the most of the passenger customers are business travellers. Most of the business travellers are derived from the age range from 20-60 years old. Such customer group are the best revenue stream to the company. It is make sense that majority of the Business travellers are loyalty to the company as compared to the Personal travellers. Among the airline class provided by the company, which is includes "Business","Eco", and "Eco-Plus", "Business" class is the most selling product class, followed by the "Eco" class, then end with "Eco-Plus". Most of the customers are dissatisfied with the services in average flight distance about 2000km mileage. Perhaps, it is due to the reason that most of the customers are demanding similar mileage distance and travel places, which in hence overloading the company capability to fulfill customer needs during that peak demand on those circumstances. Besides, based on the feature importance of all the feature ratings as comparison towards the customers' satisfaction, we may conclude that the top 3 most important factor is the inflight entertainment, followed by the comfort seat service and how easier the online booking services provided by the Airline. Hence, companies should take serious to such three factors as they want to enhance their customer satisfactions.
# 
# In addition, based on the comparison of these three models, XGBoost classifier model is the best model with consistent high scores of AUC, precision, recall, f1-score and also the accuracy score. The overall scores of all metrics are ranging from as low as 94.3% up to 95.1%. Such metric scores are considered higher as comparison to Decision Tree model.


