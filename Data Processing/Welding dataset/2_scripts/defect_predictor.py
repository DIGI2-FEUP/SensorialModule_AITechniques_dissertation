# importing libraries
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import csv

# fixing matplotlib freeze
import matplotlib

matplotlib.use('TkAgg')

dataset = 2

if dataset == 1:
    t1 = 0.6
    t2 = 1.2
elif dataset == 2:
    t1 = 1.2
    t2 = 1.5
elif dataset == 3:
    t1 = 1.5
    t2 = 1.5

def print_model(model, matriz_conf):
    print("Results for " + model)

    print(matriz_conf)

    val_precision = matriz_conf[1][1] / (matriz_conf[1][1] + matriz_conf[0][1])
    val_recall = matriz_conf[1][1] / (matriz_conf[1][1] + matriz_conf[1][0])
    val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall)

    print("Precision:", str(val_precision),
          "\nRecall:", str(val_recall),
          "\nF1 Score:", str(val_f1))


# reading data
data = pd.read_excel("../1_data/dataset" + str(dataset) + ".xlsx")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

'''
DEFECT LABEL
'''

data["Defect"] = 0
n_defects = 0

for i in data.index:
    if data["Penetration"][i] < 0.1 or data["Weld_Width"][i] < 0.5 or data["Undercut"][i] > 0.25 * t1 or data["Root_Concavity"][i] > 0.25 * t2:
        data["Defect"][i] = 1
        n_defects += 1

print(f"Amostras defeituosas: {n_defects}.")

data["Defect"] = data["Defect"].astype('category')
data["Defect"] = data["Defect"].cat.codes

'''
VISUALISATION ANALYSIS
'''
'''
cols = data.columns.tolist()


# Define the directory where you want to save the file
save_directory = "../4_figures/dataset" + str(dataset) + "/defects"
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
    plt.savefig(save_directory + "/scatter_" + str(c) + ".png")
# plt.show()

plt.figure()
plt.clf()
ax = sns.countplot(x="Defect", data=data)
plt.title("Defect")
plt.ylabel("Defect")
plt.savefig(save_directory + "/scatter_Defect.png")

# boxplots to identify outliers in continuous features
for i, c in enumerate(cols):
    plt.figure(i)
    plt.clf()
    sns.boxplot(data[c])
    plt.title(c)
    plt.ylabel(c)
    plt.savefig(save_directory + "/boxplot_" + str(c) + ".png")
# plt.show()
'''

'''
AUTO-CORRELATION ANALYSIS
'''
'''
cols = data.columns.tolist()

# Define the directory where you want to save the file
save_directory = "../4_figures/dataset" + str(dataset) + "/defects"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for i, c in enumerate(cols):
    plt.figure(i)
    plt.clf()
    ax = plot_acf(data[c], title=c)
    # plt.show()
    plt.ylabel(c)
    plt.savefig(save_directory + "/autocorrelation_" + str(c) + ".png")
'''

'''
PEARSON'S CORRELATION ANALYSIS PRE
'''
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

# saving seaborn plot
# Define the directory where you want to save the file
save_directory = "../4_figures/dataset" + str(dataset) + "/defects"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
fig = heatmap_plot.get_figure()
fig.savefig(save_directory + "/corrmatrix.png")
'''

'''
MODELLING AND VALIDATION - Defect = function( power, focus, speed )
'''

# Splitting features (X) and target (y1)
drop_columns = ["Penetration", "Weld_Width", "Undercut", "Root_Concavity",
                "Excessive_Penetration", "Excess_Weld_Metal", "Defect"]
X = data.drop(drop_columns, axis=1)
y = data["Defect"]

# Scale features
X = X.values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)
# print(X_scaled)

# train / test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

n_defects = 0
print(type(y_test))
for index, value in y_test.items():
    if value == 1:
        n_defects += 1

print(f"Defeitos em y_test: {n_defects}.")

knn_classifier = KNeighborsClassifier()
svm_classifier = SVC(random_state=42)
ridge_regressor = RidgeClassifier(random_state=42)
rf_classifier = RandomForestClassifier(random_state=42)
xgb_classifier = xgb.XGBClassifier(random_state=42)

knn_classifier.fit(x_train, y_train)
y_pred = knn_classifier.predict(x_test)
matriz_conf = confusion_matrix(y_test, y_pred)
print_model("KNN", matriz_conf)

svm_classifier.fit(x_train, y_train)
y_pred = svm_classifier.predict(x_test)
matriz_conf = confusion_matrix(y_test, y_pred)
print_model("SVM", matriz_conf)

ridge_regressor.fit(x_train, y_train)
y_pred = ridge_regressor.predict(x_test)
matriz_conf = confusion_matrix(y_test, y_pred)
print_model("Ridge", matriz_conf)

rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)
matriz_conf = confusion_matrix(y_test, y_pred)
print_model("RF", matriz_conf)

xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)
matriz_conf = confusion_matrix(y_test, y_pred)
print_model("XGB", matriz_conf)


