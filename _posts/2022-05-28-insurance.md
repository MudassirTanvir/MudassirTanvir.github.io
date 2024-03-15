---
layout: post
title: Predicting Insurance Charges Using ML
image: "/posts/insurance_predictor_theme_image.png"
tags: [Insurance Premium Charges, Machine Learning, Regression, Python]
---

Following the footsteps of data enrichment practices, the aim is to build a robust predictive model that accurately anticipates insurance charges, providing valuable insights for informed decision-making in the insurance domain.

# Table of contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth/Next Steps](#overview-growth)
    - [Key Definition](#overview-definition)
- [01. Data Overview](#data-overview)
- [02. Modelling Overview](#modelling-overview)
- [03. Linear Regression](#linreg-title)
- [04. Decision Tree](#regtree-title)
- [05. Random Forest](#rf-title)
- [06. Modelling Summary](#modelling-summary)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>


In a project focused on insurance analysis, a predictive model is being developed to gain insights into policyholder behavior and anticipate future trends. The initial dataset contains various key metrics, although there is variability in the availability of certain metrics across the dataset.

To address this variability, the aim is to build a predictive model that ctt gaccurately analyze the available metrics and provide valuable insights into insurance dynamics. By leveraging this model, a comprehensive understanding of the datasetn custome can be gained, leading to more informed decisions and improved analytical capabilities in the insurance domain.
<br>
<br>
### Actions <a name="overview-actions"></a>

We firstly needed to compile the necessary data from tables in the database, gathering key metrics that may help predict *charges*.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest
<br>
<br>

### Results <a name="overview-results"></a>

Our testing found that the Random Forest had the highest predictive accuracy.

<br>
**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.86
* Decision Tree = 0.84
* Linear Regression = 0.74

<br>
**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.83
* Decision Tree = 0.84
* Linear Regression = 0.74

As the most important outcome for this project was predictive accuracy, rather than explicitly understanding weighted drivers of prediction, we chose the Random Forest as the model to use for making predictions on the *charges* metric.
<br>
<br>
### Growth/Next Steps <a name="overview-growth"></a>

While predictive accuracy was relatively high - other modelling approaches could be tested, especially those somewhat similar to Random Forest, for example XGBoost, LightGBM to see if even more accuracy could be gained.

From a data point of view, further variables could be collected, and further feature engineering could be undertaken to ensure that we have as much useful information available for predicting customer loyalty
<br>
<br>
### Key Definition  <a name="overview-definition"></a>

The *charges* metric represents the predicted insurance premiums based on customer attributes such as age, sex, smoking status, and BMI. It quantifies the estimated amount a customer is expected to pay for insurance coverage, utilizing data-driven models to forecast charges tailored to individual characteristics.  

Example 1: Customer Y, a 35-year-old non-smoking male with a BMI of 25, resides in Region A. Based on these attributes, the predicted annual insurance premium *charges* is $1500.

Example 2: Customer Z, a 45-year-old smoking female with a BMI of 30, resides in Region B. Based on these attributes, the predicted annual insurance premium *charges* is $2500.
<br>
<br>
___

# Data Overview  <a name="data-overview"></a>

We will be creating a model to predict the *charges* metric. 



```python

# import required packages
import pandas as pd
import pickle

# import required data tables
df = ...


<br>
<br>

| **Variable Name** | **Variable Type** | **Description** |
|---|---|---|
| charges  | Dependent   | Indicates the annual insurance expense incurred by a customer |
|    age   | Independent | Indicates the age of the customer |
|    sex   | Independent | The gender provided by the customer |
|    bmi   | Independent | Indicates the bmi of the customer |
| children | Independent | Indicates the number of children the customer has |
|  smoker  | Independent | Indicates the smoking status of the customer |
|  region  | Independent | Indicates the region where the customer resides in |

___
<br>
# Modelling Overview

We will build a model that looks to accurately predict the “charges” metric for the customers, based upon the customer metrics listed above.

If that can be achieved, we can use this model to predict the customer charges for the the new customers.

As we are predicting a numeric output, we tested three regression modelling approaches, namely:

* Linear Regression
* Decision Tree
* Random Forest

___
<br>
# Linear Regression <a name="linreg-title"></a>

We utlise the scikit-learn library within Python to model our data using Linear Regression. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="linreg-import"></a>

We ensure that our data is shuffled.

```python

# import required packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

# import modelling data
df = pd.read_csv("insurance_csv")

# drop uneccessary columns
df.drop("customer_id", axis = 1, inplace = True)

# shuffle data
df = shuffle(df, random_state = 42)

```
<br>
### Data Preprocessing <a name="linreg-preprocessing"></a>

For Linear Regression we have certain data preprocessing steps that need to be addressed, including:

* Missing values in the data
* The effect of outliers
* Encoding categorical variables to numeric form
* Multicollinearity & Feature Selection

<br>
##### Missing Values

Checking for missing values and fortunately we have none.

```python

# remove rows where values are missing
df.isna().sum()

```

<br>
##### Outliers

The ability for a Linear Regression model to generalise well across *all* data can be hampered if there are outliers present.  There is no right or wrong way to deal with outliers, but it is always something worth very careful consideration - just because a value is high or low, does not necessarily mean it should not be there!

In this code section, we use **.describe()** from Pandas to investigate the spread of values for each of our predictors.  The results of this can be seen in the table below.

<br>

| **metric** | **age** | **bmi** |  **children** | **charges** | 
|----|----|----|----|----|
    mean    39.20  30.66   1.09  13270.42
    std     14.04  6.09    1.20  12110.01
    min     18.00  15.96   0.00  1121.87
    25%     27.00  26.29   0.00  4740.28
    50%     39.00  30.40   1.00  9382.03
    75%     51.00  34.69   2.00  16639.91
    max     64.00  53.13   5.00  63770.42

<br>
Based on this investigation, we see some *max* column values for the some variables to be much higher than the *median* value.

This is for columns *bmi*


Because of this, we apply some outlier removal in order to facilitate generalisation across the full dataset.

We do this using the "boxplot approach" where we remove any rows where the values within those columns are outside of the interquartile range multiplied by 2.

<br>
```python

outlier_investigation = df.describe()
outlier_columns = ["bmi", "age"]

# boxplot approach
for column in outlier_columns:
    
    lower_quartile = df[column].quantile(0.25)
    upper_quartile = df[column].quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 2
    min_border = lower_quartile - iqr_extended
    max_border = upper_quartile + iqr_extended
    
    outliers = df[(df[column] < min_border) | (df[column] > max_border)].index
    print(f"{len(outliers)} outliers detected in column {column}")
    
    df.drop(outliers, inplace = True)

```

<br>
##### Split Out Data For Modelling

In the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# split data into X and y objects for modelling
X = df.drop(["charges"], axis = 1)
y = df["charges"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have three categorical variable one being *gender* which has values of "M" for Male, "F" for Female and other being *smoker* which has values of "yes" for customer who smoke, "no for customer who don't smoke and smiliarly for the *region* we have respective values.

The Linear Regression algorithm can't deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *sex*, *smoker* and *region*  doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

One Hot Encoding can be thought of as a way to represent categorical variables as binary vectors, in other words, a set of *new* columns for each categorical value with either a 1 or a 0 saying whether that value is true or not for that observation.  These new columns would go into our model as input variables, and the original column is discarded.

We also drop one of the new columns using the parameter *drop = "first"*.  We do this to avoid the *dummy variable trap* where our newly created encoded columns perfectly predict each other - and we run the risk of breaking the assumption that there is no multicollinearity, a requirement or at least an important consideration for some models, Linear Regression being one of them! Multicollinearity occurs when two or more input variables are *highly* correlated with each other, it is a scenario we attempt to avoid as in short, while it won't neccessarily affect the predictive accuracy of our model, it can make it difficult to trust the statistics around how well the model is performing, and how much each input variable is truly having.

In the code, we also make sure to apply *fit_transform* to the training set, but only *transform* to the test set.  This means the One Hot Encoding logic will *learn and apply* the "rules" from the training data, but only *apply* them to the test data.  This is important in order to avoid *data leakage* where the test set *learns* information about the training data, and means we can't fully trust model performance metrics!

For ease, after we have applied One Hot Encoding, we turn our training and test objects back into Pandas Dataframes, with the column names applied.

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["sex", "smoker", "region"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
##### Feature Selection

Feature Selection is the process used to select the input variables that are most important to your Machine Learning task.  It can be a very important addition or at least, consideration, in certain scenarios.  The potential benefits of Feature Selection are:

* **Improved Model Accuracy** - eliminating noise can help true relationships stand out
* **Lower Computational Cost** - our model becomes faster to train, and faster to make predictions
* **Explainability** - understanding & explaining outputs for stakeholder & customers becomes much easier

There are many, many ways to apply Feature Selection.  These range from simple methods such as a *Correlation Matrix* showing variable relationships, to *Univariate Testing* which helps us understand statistical relationships between variables, and then to even more powerful approaches like *Recursive Feature Elimination (RFE)* which is an approach that starts with all input variables, and then iteratively removes those with the weakest relationships with the output variable.

For our task we applied a variation of Reursive Feature Elimination called *Recursive Feature Elimination With Cross Validation (RFECV)* where we split the data into many "chunks" and iteratively trains & validates models on each "chunk" seperately.  This means that each time we assess different models with different variables included, or eliminated, the algorithm also knows how accurate each of those models was.  From the suite of model scenarios that are created, the algorithm can determine which provided the best accuracy, and thus can infer the best set of input variables to use!

<br>
```python

# instantiate RFECV & the model type to be utilised
regressor = LinearRegression()
feature_selector = RFECV(regressor)

# fit RFECV onto our training & test data
fit = feature_selector.fit(X_train,y_train)

# extract & print the optimal number of features
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

# limit our training & test sets to only include the selected variables
X_train = X_train.loc[:, feature_selector.get_support()]
X_test = X_test.loc[:, feature_selector.get_support()]

```

<br>
The below code then produces a plot that visualises the cross-validated accuracy with each potential number of features

```python

plt.style.use('seaborn-poster')
plt.plot(range(1, len(fit.cv_results_['mean_test_score']) + 1), fit.cv_results_['mean_test_score'], marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.cv_results_['mean_test_score']),4)})")
plt.tight_layout()


```

<br>
This creates the below plot, which shows us that the highest cross-validated accuracy (0.7358) is actually when we include all eight of our original input variables.  This is higher than 6 included variables, and 7 included variables.  We will continue on with all 8!

<br>
![alt text](/img/posts/insurance_lin-reg-feature-selection-plot.png "Linear Regression Feature Selection Plot")

<br>
### Model Training <a name="linreg-model-training"></a>

Instantiating and training our Linear Regression model is done using the below code

```python

# instantiate our model object
regressor = LinearRegression()

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="linreg-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>
##### Calculate R-Squared

R-Squared is a metric that shows the percentage of variance in our output variable *y* that is being explained by our input variable(s) *x*.  It is a value that ranges between 0 and 1, with a higher value showing a higher level of explained variance.  Another way of explaining this would be to say that, if we had an r-squared score of 0.8 it would suggest that 80% of the variation of our output variable is being explained by our input variables - and something else, or some other variables must account for the other 20%

To calculate r-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting r-squared score from this is **0.74**

<br>
##### Calculate Cross Validated R-Squared

An even more powerful and reliable way to assess model performance is to utilise Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results - and we can take the average of these to give a much more robust & reliable view of how our model will perform on new, un-seen data!

In the code below, we put this into place.  We first specify that we want 4 "chunks" and then we pass in our regressor object, training set, and test set.  We also specify the metric we want to assess with, in this case, we stick with r-squared.

Finally, we take a mean of all four test set results.

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

The mean cross-validated r-squared score from this is **0.738**

<br>
##### Calculate Adjusted R-Squared

When applying Linear Regression with *multiple* input variables, the r-squared metric on it's own *can* end up being an overinflated view of goodness of fit.  This is because each input variable will have an *additive* effect on the overall r-squared score.  In other words, every input variable added to the model *increases* the r-squared value, and *never decreases* it, even if the relationship is by chance.  

**Adjusted R-Squared** is a metric that compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.  It is best practice to use Adjusted R-Squared when assessing the results of a Linear Regression with multiple input variables, as it gives a fairer perception the fit of the data.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting *adjusted* r-squared score from this is **0.739** which as expected, is slightly lower than the score we got for r-squared on it's own.

<br>
### Model Summary Statistics <a name="linreg-model-summary"></a>

Although our overall goal for this project is predictive accuracy, rather than an explcit understanding of the relationships of each of the input variables and the output variable, it is always interesting to look at the summary statistics for these.
<br>
```python

# extract model coefficients
coefficients = pd.DataFrame(regressor.coef_)
input_variable_names = pd.DataFrame(X_train.columns)
summary_stats = pd.concat([input_variable_names,coefficients], axis = 1)
summary_stats.columns = ["input_variable", "coefficient"]

# extract model intercept
regressor.intercept_

```
<br>
The information from that code block can be found in the table below:
<br>

| **input_variable** | **coefficient** |
|---|---|
| age | 256.20 |
| bmi | 362.72 |
| children | 541.87 |
| sex_male | -58.45 |
| smoker_yes | 23346.29 |
| region_northwest | -979.48 |
| region_southeast | -1467.70 |
| region_southwest | -1529.61 |

<br>
The coefficient value for each of the input variables, along with that of the intercept would make up the equation for the line of best fit for this particular model (or more accurately, in this case it would be the plane of best fit, as we have multiple input variables).

For each input variable, the coefficient value we see above tells us, with *everything else staying constant* how many units the output variable (charges) would change with a *one unit change* in this particular input variable.

In the table provided, the coefficient for the *age* input variable stands at 256.20. This indicates that for every additional year a customer ages, there is a corresponding increase of 256.20 in their insurance *charge*. This observation aligns with our intuition, as older customers typically incur higher insurance costs due to the increased complexities associated with aging

___
<br>
# Decision Tree <a name="regtree-title"></a>

We will again utlise the scikit-learn library within Python to model our data using a Decision Tree. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="regtree-import"></a>

Since we saved our modelling data as a pickle file, we import it.  We ensure we remove the id column, and we also ensure our data is shuffled.

```python

# import required packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

# import modelling data
df = pd.read_csv("data/insurance_charges.csv")


# shuffle data
df = shuffle(df, random_state = 42)

```
<br>
### Data Preprocessing <a name="regtree-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Decision Trees are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

Fortunately we have no missing values so we are good to go.

```python

# remove rows where values are missing
df.isna().sum()
```

<br>
##### Split Out Data For Modelling

In exactly the same way we did for Linear Regression, in the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# split data into X and y objects for modelling
X = df.drop(["charges"], axis = 1)
y = df["charges"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have three categorical variable one being *gender* which has values of "M" for Male, "F" for Female and other being *smoker* which has values of "yes" for customer who smoke, "no for customer who don't smoke and smiliarly for the *region* we have respective values.

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Decision Trees are not, so the required preprocessing here is a bit lighter. However, logic will be put in place for:

Missing values in the data
Encoding categorical variables to numeric form

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["sex", "smoker", "region"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training <a name="regtree-model-training"></a>

Instantiating and training our Decision Tree model is done using the below code.  We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

```python

# instantiate our model object
regressor = DecisionTreeRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="regtree-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *charges* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>
##### Calculate R-Squared

To calculate r-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting r-squared score from this is **0.64**

<br>
##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression, we will again utilise Cross Validation.

Instead of simply dividing our data into a single training set, and a single test set, with Cross Validation we break our data into a number of "chunks" and then iteratively train the model on all but one of the "chunks", test the model on the remaining "chunk" until each has had a chance to be the test set.

The result of this is that we are provided a number of test set validation results - and we can take the average of these to give a much more robust & reliable view of how our model will perform on new, un-seen data!

In the code below, we put this into place.  We again specify that we want 4 "chunks" and then we pass in our regressor object, training set, and test set.  We also specify the metric we want to assess with, in this case, we stick with r-squared.

Finally, we take a mean of all four test set results.

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

The mean cross-validated r-squared score from this is **0.66** which is lower than we saw for Linear Regression.

<br>
##### Calculate Adjusted R-Squared

Just like we did with Linear Regression, we will also calculate the *Adjusted R-Squared* which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting *adjusted* r-squared score from this is **0.63** which as expected, is slightly lower than the score we got for r-squared on it's own.

<br>
### Decision Tree Regularisation <a name="regtree-model-regularisation"></a>

Decision Tree's can be prone to over-fitting, in other words, without any limits on their splitting, they will end up learning the training data perfectly.  We would much prefer our model to have a more *generalised* set of rules, as this will be more robust & reliable when making predictions on *new* data.

One effective method of avoiding this over-fitting, is to apply a *max depth* to the Decision Tree, meaning we only allow it to split the data a certain number of times before it is required to stop.

Unfortunately, we don't necessarily know the *best* number of splits to use for this - so below we will loop over a variety of values and assess which gives us the best predictive performance!

<br>
```python

# finding the best max_depth

# set up range for search, and empty list to append accuracy scores to
max_depth_list = list(range(1,10))
accuracy_scores = []

# loop through each possible depth, train and validate model, append test set accuracy
for depth in max_depth_list:
    
    regressor = DecisionTreeRegressor(max_depth = depth, random_state = 42)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    accuracy_scores.append(accuracy)
    
# store max accuracy, and optimal depth    
max_accuracy = max(accuracy_scores)
max_accuracy_idx = accuracy_scores.index(max_accuracy)
optimal_depth = max_depth_list[max_accuracy_idx]

# plot accuracy by max depth
plt.plot(max_depth_list,accuracy_scores)
plt.scatter(optimal_depth, max_accuracy, marker = "x", color = "red")
plt.title(f"Accuracy by Max Depth \n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,4)})")
plt.xlabel("Max Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()

```
<br>
That code gives us the below plot - which visualises the results!

<br>
![alt text](/img/posts/insurance_regression-tree-max-depth-plot.png "Decision Tree Max Depth Plot")

<br>
In the plot we can see that the *maximum* classification accuracy on the test set is found when applying a *max_depth* value of 3.

<br>
### Visualise Our Decision Tree <a name="regtree-visualise"></a>

To see the decisions that have been made in the (re-fitted) tree, we can use the plot_tree functionality that we imported from scikit-learn.  To do this, we use the below code:

<br>
```python

# re-fit our model using max depth of 4
regressor = DecisionTreeRegressor(random_state = 42, max_depth = 4)
regressor.fit(X_train, y_train)

# plot the nodes of the decision tree
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 16)

```
<br>
That code gives us the below plot:

<br>
![alt text](/img/posts/insurance_regression-tree-nodes-plot.png "Decision Tree Max Depth Plot")

<br>
This is a very powerful visual, and one that can be shown to stakeholders in the business to ensure they understand exactly what is driving the predictions.

One interesting thing to note is that the *very first split* appears to be using the variable *distance from store* so it would seem that this is a very important variable when it comes to predicting loyalty!

___
<br>
# Random Forest <a name="rf-title"></a>

We will again utlise the scikit-learn library within Python to model our data using a Random Forest. The code sections below are broken up into 4 key sections:

* Data Import
* Data Preprocessing
* Model Training
* Performance Assessment

<br>
### Data Import <a name="rf-import"></a>

We import the data and the required packages.

```python

# import required packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

# import modelling data
df = pd.read_csv("insurance_charges.csv")


# shuffle data
df = shuffle(df, random_state = 42)

```
<br>
### Data Preprocessing <a name="rf-preprocessing"></a>

While Linear Regression is susceptible to the effects of outliers, and highly correlated input variables - Random Forests, just like Decision Trees, are not, so the required preprocessing here is lighter. We still however will put in place logic for:

* Missing values in the data
* Encoding categorical variables to numeric form

<br>
##### Missing Values

Fortunately we have no missing values so we are good to go.

```python

# remove rows where values are missing
df.isna().sum()
```

<br>
##### Split Out Data For Modelling

In exactly the same way we did for Linear Regression, in the next code block we do two things, we firstly split our data into an **X** object which contains only the predictor variables, and a **y** object that contains only our dependent variable.

Once we have done this, we split our data into training and test sets to ensure we can fairly validate the accuracy of the predictions on data that was not used in training.  In this case, we have allocated 80% of the data for training, and the remaining 20% for validation.

<br>
```python

# split data into X and y objects for modelling
X = df.drop(["charges"], axis = 1)
y = df["charges"]

# split out training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

```

<br>
##### Categorical Predictor Variables

In our dataset, we have three categorical variable one being *gender* which has values of "M" for Male, "F" for Female and other being *smoker* which has values of "yes" for customer who smoke, "no" for customer who don't smoke and smiliarly for the *region* we have respective values.

Just like the Linear Regression algorithm, Random Forests cannot deal with data in this format as it can't assign any numerical meaning to it when looking to assess the relationship between the variable and the dependent variable.

As *sex*, *smoker* and *region*  doesn't have any explicit *order* to it, in other words, Male isn't higher or lower than Female and vice versa - one appropriate approach is to apply One Hot Encoding to the categorical column.

<br>
```python

# list of categorical variables that need encoding
categorical_vars = ["sex", "smoker", "region"]

# instantiate OHE class
one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

# apply OHE
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

# extract feature names for encoded columns
encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

# turn objects back to pandas dataframe
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

```

<br>
### Model Training <a name="rf-model-training"></a>

Instantiating and training our Random Forest model is done using the below code.  We use the *random_state* parameter to ensure we get reproducible results, and this helps us understand any improvements in performance with changes to model hyperparameters.

We leave the other parameters at their default values, meaning that we will just be building 100 Decision Trees in this Random Forest.

```python

# instantiate our model object
regressor = RandomForestRegressor(random_state = 42)

# fit our model using our training & test sets
regressor.fit(X_train, y_train)

```

<br>
### Model Performance Assessment <a name="rf-model-assessment"></a>

##### Predict On The Test Set

To assess how well our model is predicting on new data - we use the trained model object (here called *regressor*) and ask it to predict the *loyalty_score* variable for the test set

```python

# predict on the test set
y_pred = regressor.predict(X_test)

```

<br>
##### Calculate R-Squared

To calculate r-squared, we use the following code where we pass in our *predicted* outputs for the test set (y_pred), as well as the *actual* outputs for the test set (y_test)

```python

# calculate r-squared for our test set predictions
r_squared = r2_score(y_test, y_pred)
print(r_squared)

```

The resulting r-squared score from this is **0.865** - higher than both Linear Regression & the Decision Tree.

<br>
##### Calculate Cross Validated R-Squared

As we did when testing Linear Regression & our Decision Tree, we will again utilise Cross Validation (for more info on how this works, please refer to the Linear Regression section above)

```python

# calculate the mean cross validated r-squared for our test set predictions
cv = KFold(n_splits = 4, shuffle = True, random_state = 42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv = cv, scoring = "r2")
cv_scores.mean()

```

The mean cross-validated r-squared score from this is **0.82** which again is higher than we saw for both Linear Regression & our Decision Tree.

<br>
##### Calculate Adjusted R-Squared

Just like we did with Linear Regression & our Decision Tree, we will also calculate the *Adjusted R-Squared* which compensates for the addition of input variables, and only increases if the variable improves the model above what would be obtained by probability.

```python

# calculate adjusted r-squared for our test set predictions
num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars - 1)
print(adjusted_r_squared)

```

The resulting *adjusted* r-squared score from this is **0.860** which as expected, is slightly lower than the score we got for r-squared on it's own - but again higher than for our other models.

<br>
### Feature Importance <a name="rf-model-feature-importance"></a>

In our Linear Regression model, to understand the relationships between input variables and our ouput variable, loyalty score, we examined the coefficients.  With our Decision Tree we looked at what the earlier splits were.  These allowed us some insight into which input variables were having the most impact.

Random Forests are an ensemble model, made up of many, many Decision Trees, each of which is different due to the randomness of the data being provided, and the random selection of input variables available at each potential split point.

Because of this, we end up with a powerful and robust model, but because of the random or different nature of all these Decision trees - the model gives us a unique insight into how important each of our input variables are to the overall model.  

As we’re using random samples of data, and input variables for each Decision Tree - there are many scenarios where certain input variables are being held back and this enables us a way to compare how accurate the models predictions are if that variable is or isn’t present.

So, at a high level, in a Random Forest we can measure *importance* by asking *How much would accuracy decrease if a specific input variable was removed or randomised?*

If this decrease in performance, or accuracy, is large, then we’d deem that input variable to be quite important, and if we see only a small decrease in accuracy, then we’d conclude that the variable is of less importance.

At a high level, there are two common ways to tackle this.  The first, often just called **Feature Importance** is where we find all nodes in the Decision Trees of the forest where a particular input variable is used to split the data and assess what the Mean Squared Error (for a Regression problem) was before the split was made, and compare this to the Mean Squared Error after the split was made.  We can take the *average* of these improvements across all Decision Trees in the Random Forest to get a score that tells us *how much better* we’re making the model by using that input variable.

If we do this for *each* of our input variables, we can compare these scores and understand which is adding the most value to the predictive power of the model!

The other approach, often called **Permutation Importance** cleverly uses some data that has gone *unused* at when random samples are selected for each Decision Tree (this stage is called "bootstrap sampling" or "bootstrapping")

These observations that were not randomly selected for each Decision Tree are known as *Out of Bag* observations and these can be used for testing the accuracy of each particular Decision Tree.

For each Decision Tree, all of the *Out of Bag* observations are gathered and then passed through.  Once all of these observations have been run through the Decision Tree, we obtain an accuracy score for these predictions, which in the case of a regression problem could be Mean Squared Error or r-squared.

In order to understand the *importance*, we *randomise* the values within one of the input variables - a process that essentially destroys any relationship that might exist between that input variable and the output variable - and run that updated data through the Decision Tree again, obtaining a second accuracy score.  The difference between the original accuracy and the new accuracy gives us a view on how important that particular variable is for predicting the output.

*Permutation Importance* is often preferred over *Feature Importance* which can at times inflate the importance of numerical features. Both are useful, and in most cases will give fairly similar results.

Let's put them both in place, and plot the results...

<br>
```python

# calculate feature importance
feature_importance = pd.DataFrame(regressor.feature_importances_)
feature_names = pd.DataFrame(X.columns)
feature_importance_summary = pd.concat([feature_names,feature_importance], axis = 1)
feature_importance_summary.columns = ["input_variable","feature_importance"]
feature_importance_summary.sort_values(by = "feature_importance", inplace = True)

# plot feature importance
plt.barh(feature_importance_summary["input_variable"],feature_importance_summary["feature_importance"])
plt.title("Feature Importance of Random Forest")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# calculate permutation importance
result = permutation_importance(regressor, X_test, y_test, n_repeats = 10, random_state = 42)
permutation_importance = pd.DataFrame(result["importances_mean"])
feature_names = pd.DataFrame(X.columns)
permutation_importance_summary = pd.concat([feature_names,permutation_importance], axis = 1)
permutation_importance_summary.columns = ["input_variable","permutation_importance"]
permutation_importance_summary.sort_values(by = "permutation_importance", inplace = True)

# plot permutation importance
plt.barh(permutation_importance_summary["input_variable"],permutation_importance_summary["permutation_importance"])
plt.title("Permutation Importance of Random Forest")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

```
<br>
That code gives us the below plots - the first being for *Feature Importance* and the second for *Permutation Importance*!

<br>
![alt text](/img/posts/insurance_rf-regression-feature-importance.png "Random Forest Feature Importance Plot")
<br>
<br>
![alt text](/img/posts/insurance_rf-regression-permutation-importance.png "Random Forest Permutation Importance Plot")

<br>
The overall story from both approaches is very similar, in that by far, the most important or impactful input variable is *smoker_yes* which is the same insights we derived when assessing our Linear Regression & Decision Tree models.

There are slight differences in the order or "importance" for the remaining variables but overall they have provided similar findings.

___
<br>
# Modelling Summary  <a name="modelling-summary"></a>

The most important outcome for this project was predictive accuracy, rather than explicitly understanding the drivers of prediction. Based upon this, we chose the model that performed the best when predicted on the test set - the Random Forest.

<br>
**Metric 1: Adjusted R-Squared (Test Set)**

* Random Forest = 0.865
* Decision Tree = 0.849
* Linear Regression = 0.754

<br>
**Metric 2: R-Squared (K-Fold Cross Validation, k = 4)**

* Random Forest = 0.860
* Decision Tree = 0.841
* Linear Regression = 0.738

<br>
Even though we were not specifically interested in the drivers of prediction, it was interesting to see across all three modelling approaches, that the input variable with the biggest impact on the prediction was *smoker_yes*

<br>
