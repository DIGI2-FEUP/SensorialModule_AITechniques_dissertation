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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import csv

# fixing matplotlib freeze
import matplotlib

matplotlib.use('TkAgg')

dataset = 3

# reading data
data = pd.read_excel("../1_data/dataset" + str(dataset) + ".xlsx")

pd.set_option('display.max_columns', None)

# Splitting features (X) and target (y1)
drop_columns = ["Undercut", "Root_Concavity",
                "Excessive_Penetration", "Excess_Weld_Metal"]
data = data.drop(drop_columns, axis=1)

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
save_directory = "../4_figures"
# Check if the directory exists; if not, create it
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
fig = heatmap_plot.get_figure()
fig.savefig(save_directory + "/corrmatrix_" + str(dataset) + ".png")