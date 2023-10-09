# importing libraries
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
import csv

# fixing matplotlib freeze
import matplotlib

matplotlib.use('TkAgg')

dataset = 3

# reading data
data = pd.read_excel("../1_data/dataset" + str(dataset) + ".xlsx")

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

# print(data)


'''
MODELLING AND VALIDATION - penetration = function( power, focus, speed )
'''

# Splitting features (X) and target (y1)
drop_columns = ["Penetration", "Weld_Width", "Undercut", "Root_Concavity",
                "Excessive_Penetration", "Excess_Weld_Metal"]
X = data.drop(drop_columns, axis=1)

# Scale features
X = X.values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled)


# .csv file to store the results
file_path = '../5_results/best_models.csv'
if not os.path.exists(file_path):
    columns = ["dataset", "target", "model", "r2", "mse", "rmse"]
    # Open the CSV file in append mode and write the new data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)


# SAVE DATASET
data.to_csv("../1_data/final_clean_dataset_d{}.csv".format(dataset), index=False)



### PENETRATION
y = data["Penetration"]

print("\n###########\nPENETRATION\n###########")

# train / test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save test dataset
idx_list = x_test.index.tolist()
test_dataset = data.iloc[idx_list]
test_dataset.to_csv("../1_data/test_dataset_d{}.csv".format(dataset), index=False)

## keeping model with best r2 in memory
best_model = ""
best_r2 = 0

knn_regressor = KNeighborsRegressor()
svm_regressor = SVR()
rf_regressor = RandomForestRegressor()
xgb_regressor = xgb.XGBRegressor()

models = [
    ("K-Nearest Neighbors", knn_regressor),
    ("Support Vector Machine", svm_regressor),
    ("Random Forest", rf_regressor),
    ("XGBoost", xgb_regressor)
]

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    score = model.score(x_test, y_test)
    print(f"\n{name} R² score: {score:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    new_data = [dataset, "Penetration", name, score, mse, rmse]

    # Open the CSV file in append mode and write the new data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_data)

    if score > best_r2:

        print("Best Model: " + name + " with score: {}".format(score))

        best_r2 = score
        best_model = name

        # saving model
        filename = '../3_models/d{}_penetration.sav'.format(dataset)
        pickle.dump(model, open(filename, 'wb'))


### WELD_WIDTH
y = data["Weld_Width"]

print("\n\n\n##########\nWELD_WIDTH\n########## ")

## keeping model with best r2 in memory
best_model = ""
best_r2 = 0

knn_regressor = KNeighborsRegressor()
svm_regressor = SVR()
rf_regressor = RandomForestRegressor()
xgb_regressor = xgb.XGBRegressor()

models = [
    ("K-Nearest_Neighbors", knn_regressor),
    ("Support_Vector_Machine", svm_regressor),
    ("Random_Forest", rf_regressor),
    ("XGBoost", xgb_regressor)
]

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    score = model.score(x_test, y_test)
    print(f"\n{name} R² score: {score:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    new_data = [dataset, "Weld_Width", name, score, mse, rmse]

    # Open the CSV file in append mode and write the new data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(new_data)

    if score > best_r2:

        print("Best Model: " + name + " with score: {}".format(score))

        best_r2 = score
        best_model = name

        # saving model
        filename = '../3_models/d{}_weld_width.sav'.format(dataset)
        pickle.dump(model, open(filename, 'wb'))



