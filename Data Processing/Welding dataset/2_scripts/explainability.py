# loading packages
import os

import pandas as pd
import numpy as np

import shap
import xgboost
import matplotlib.pylab as plt

from sklearn import preprocessing

import pickle

# selecting dataset for analysis
dataset = 3
# feature = "Penetration"
feature = "Weld_Width"

# is best model an XGBoost algorithm?
## yes --> 1; no --> 0
is_xgb = 1

######################
######################
### OBTAINING DATA & MODEL

# dataset
filename = "../1_data/final_clean_dataset_d{}.csv".format(dataset)
data = pd.read_csv(filename)

drop_columns = ["Undercut", "Root_Concavity",
                "Excessive_Penetration", "Excess_Weld_Metal"]
if feature == "Penetration":
    drop_columns.append("Weld_Width")
else:
    drop_columns.append("Penetration")

data = data.drop(drop_columns, axis=1)

# test subset
filename = "../1_data/test_dataset_d{}.csv".format(dataset)
test_dataset = pd.read_csv(filename)
test_dataset = test_dataset.drop(drop_columns, axis=1)

# model with best recall
filename = "../3_models/d{}_{}.sav".format(dataset, feature.lower())
print(filename)
model = pickle.load(open(filename, 'rb'))

######################
######################
### DATA PREPARATION

# rearranging target column order
cols = list(test_dataset.columns.values)
cols.pop(cols.index(feature))
test_dataset = test_dataset[cols + [feature]]

cols = list(data.columns.values)
cols.pop(cols.index(feature))
data = data[cols + [feature]]

#########
# Test Dataset's input vs target features separation
if is_xgb:
    original_featnames = test_dataset.columns.tolist()
    xgb_featnames = ['0', '1', '2']

    xtest_dataset = test_dataset.iloc[:, :-1]
    xtest_dataset.columns = xgb_featnames
    ytest_dataset = test_dataset.iloc[:, -1]

else:
    xtest_dataset = test_dataset.iloc[:, :-1]
    ytest_dataset = test_dataset.iloc[:, -1]

#########
# Data Normalization step

# fitting min-max scaler to full dataset
y_values = data[feature]
x_values = data.drop(feature, axis=1)

x_values = x_values.values

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit(x_values)

# test dataset transformation
xtest_dataset = xtest_dataset.values
xtest_dataset = min_max_scaler.transform(xtest_dataset)

# retrieving dataframe format
xtest_dataset = pd.DataFrame(xtest_dataset)
xtest_dataset.columns = cols

######################
######################
### TREE-SHAP EXPLAINER

# SHAP explainer for tree-based models instantiation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(xtest_dataset)

## GRAPHS
# visualize one sample
plt.clf()
plt = shap.force_plot(explainer.expected_value, shap_values[0, :], xtest_dataset.iloc[0, :], matplotlib=True, show=False)
plt.savefig('../4_figures/explainability_dataset{}_{}_1sample.png'.format(dataset, feature))

# visualize full test dataset
plt.clf()
shap.summary_plot(shap_values, xtest_dataset, plot_type="bar", show=False)
plt.savefig('../4_figures/explainability_dataset{}_{}_full_test.png'.format(dataset, feature))

# visualize summary plot
plt.clf()
shap.summary_plot(shap_values[:100, :], xtest_dataset.iloc[:100, :], show=False)
plt.savefig('../4_figures/explainability_dataset{}_{}_k1test_summary.png'.format(dataset, feature))
