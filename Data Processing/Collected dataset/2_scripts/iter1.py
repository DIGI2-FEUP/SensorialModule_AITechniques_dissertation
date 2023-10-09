# importing libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# reading data
data = pd.read_csv("../1_data/dataset.csv")

# statistical analysis
# print(data)
# print(data.isna().sum())    # check if there is any NaN value

# remover id
data = data.drop("id", axis=1)


'''
VISUALISATION ANALYSIS
'''
'''
cols = data.columns.tolist()

# Define the directory where you want to save the file
save_directory = "../4_figures/1_preprocessing"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

x = range(len(data[:]))

# continuous features
for i, c in enumerate(cols):
    plt.figure(i)
    plt.clf()
    ax = plt.scatter(x, data[c])
    plt.title(c)
    plt.ylabel(c)
    plt.savefig("../4_figures/1_preprocessing/scatter_" + str(c) + ".png")
# plt.show()

# boxplots to identify outliers in continuous features
for i, c in enumerate(cols):
    plt.figure(i)
    plt.clf()
    sns.boxplot(data[c])
    plt.title(c)
    plt.ylabel(c)
    plt.savefig("../4_figures/1_preprocessing/boxplot_" + str(c) + ".png")
# plt.show()
'''


'''
AUTO-CORRELATION ANALYSIS
'''
'''
cols = data.columns.tolist()


# Define the directory where you want to save the file
save_directory = "../4_figures/2_autocorrelation"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for i, c in enumerate(cols):
    plt.figure(i)
    plt.clf()
    ax = plot_acf(data[c], title=c)
    # plt.show()
    plt.ylabel(c)
    plt.savefig("../4_figures/2_autocorrelation/" + str(c) + ".png")
'''


'''
PEARSON'S CORRELATION ANALYSIS PRE
'''

# Correlation matrix
corr_matrix = data.corr()

# creating mask to print only lower matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # mask keeps values == 1, on entries we want to remove from plot
# print(mask)

# custom diverging palette
cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)

plt.figure(figsize=(16, 12))

heatmap_plot = sns.heatmap(corr_matrix, mask=mask, center=0, annot=False,
                           fmt='.2f', square=True, cmap=cmap)
heatmap_plot.set_xticklabels(heatmap_plot.get_xmajorticklabels(), fontsize=20)
heatmap_plot.set_yticklabels(heatmap_plot.get_ymajorticklabels(), fontsize=20)

# saving seaborn plot
# Define the directory where you want to save the file
save_directory = "../4_figures/3_corrmatrix"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
fig = heatmap_plot.get_figure()
fig.savefig("../4_figures/3_corrmatrix/corrmatrix_pre.png")



'''
REMOVE UNECESSARY COLUMNS AND FEATURES
'''

# delta_red = data["red"].max() - data["red"].min()           # 47.82868682916282
# delta_green = data["green"].max() - data["green"].min()     # 77.97345070264029
# delta_blue = data["blue"].max() - data["blue"].min()        # 108.94904441458294
#
# print("Amplitude red: {}".format(delta_red))
# print("Amplitude green: {}".format(delta_green))
# print("Amplitude blue: {}".format(delta_blue))

# remover colunas "desnecessárias"
col_remove = ["red", "green"]
data = data.drop(col_remove, axis=1)

# print(data)


'''
PEARSON'S CORRELATION ANALYSIS POST
'''

# Correlation matrix
corr_matrix = data.corr()

# creating mask to print only lower matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # mask keeps values == 1, on entries we want to remove from plot
# print(mask)

# custom diverging palette
cmap = sns.diverging_palette(250, 15, s=75, l=40,
                             n=9, center="light", as_cmap=True)

plt.figure(figsize=(16, 12))

heatmap_plot = sns.heatmap(corr_matrix, mask=mask, center=0, annot=False,
                           fmt='.2f', square=True, cmap=cmap)
heatmap_plot.set_xticklabels(heatmap_plot.get_xmajorticklabels(), fontsize=20)
heatmap_plot.set_yticklabels(heatmap_plot.get_ymajorticklabels(), fontsize=20)

# saving seaborn plot
# Define the directory where you want to save the file
save_directory = "../4_figures/3_corrmatrix"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
fig = heatmap_plot.get_figure()
fig.savefig("../4_figures/3_corrmatrix/corrmatrix_post.png")



'''
MODELLING AND VALIDATION
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import csv

# Splitting features (X) and target (y)
X = data.drop("blue", axis=1)
y = data["blue"]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
# print(X_scaled)

# train / test split
# x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

knn_regressor = KNeighborsRegressor()
svm_regressor = SVR()
ridge_regressor = Ridge()
lasso_regressor = Lasso()
linear_regressor = LinearRegression()
rf_regressor = RandomForestRegressor()
xgb_regressor = xgb.XGBRegressor()

k = 10000

scores = []
# K-Nearest Neighbors Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    knn_regressor.fit(x_train, y_train)
    knn_score = knn_regressor.score(x_test, y_test)
    scores.append(knn_score)

knn_scores = sum(scores)/len(scores)
print("K-Nearest Neighbors average score:", sum(scores)/len(scores))

scores = []
# Support Vector Machine Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    svm_regressor.fit(x_train, y_train)
    svm_score = svm_regressor.score (x_test, y_test)
    scores.append(svm_score)

svm_scores = sum(scores)/len(scores)
print("SVM average score:", sum(scores)/len(scores))

scores = []
# Ridge Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    ridge_regressor.fit(x_train, y_train)
    ridge_score = ridge_regressor.score(x_test, y_test)
    scores.append(ridge_score)

ridge_scores = sum(scores)/len(scores)
print("Ridge average score:", sum(scores)/len(scores))

scores = []
# Lasso Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    lasso_regressor.fit(x_train, y_train)
    lasso_score = lasso_regressor.score(x_test, y_test)
    scores.append(lasso_score)

lasso_scores = sum(scores)/len(scores)
print("Lasso average score:", sum(scores)/len(scores))

scores = []
# Linear Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    linear_regressor.fit(x_train, y_train)
    linear_score = linear_regressor.score(x_test, y_test)
    scores.append(linear_score)

linear_scores = sum(scores)/len(scores)
print("Linear average score:", sum(scores)/len(scores))

scores = []
# Random Forest Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    rf_regressor.fit(x_train, y_train)
    rf_score = rf_regressor.score(x_test, y_test)
    scores.append(rf_score)

rf_scores = sum(scores)/len(scores)
print("Random Forest average score:", sum(scores)/len(scores))

scores = []
# XGBoost Regression
for i in range(k):
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    xgb_regressor.fit(x_train, y_train)
    xgb_score = xgb_regressor.score(x_test, y_test)
    scores.append(xgb_score)

xgb_scores = sum(scores)/len(scores)
print("XGBoost average score:", sum(scores)/len(scores))

# .csv file to store the results
file_path = '../3_models/results.csv'

# columns = ["k", "knn", "svm", "ridge", "lasso", "linear", "rf", "xgb"]
new_data = [k, knn_scores, svm_scores, ridge_scores, lasso_scores, linear_scores, rf_scores, xgb_scores]

# Open the CSV file in append mode and write the new data
with open(file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(new_data)
