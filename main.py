from flask import Flask,request,render_template,url_for
import numpy as np
import pandas as pd
import pickle as pkl
import json

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import mutual_info_classif,f_classif,chi2,SelectKBest
from skfeature.function.similarity_based import fisher_score

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import xgboost

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


app = Flask('__main__')
# @app.route('/')
# def connect():
#     return 'SUCCESS'

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Breast Cancer Wisconsin\artifacts\x_columns.json') as file:
    columns = json.load(file)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Breast Cancer Wisconsin\artifacts\xgb_model.pkl','rb') as file1:
    xg_model = pkl.load(file1)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Breast Cancer Wisconsin\artifacts\normal_scaler.pkl','rb') as file2:
    normal_scaler = pkl.load(file2)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Breast Cancer Wisconsin\artifacts\label_encoder.pkl','rb') as file2:
    le = pkl.load(file2)

# print(columns)
# print(le.classes_)

@app.route('/')
def render():
    return render_template('index.html')

@app.route('/input', methods=['GET','POST'])
def predict():
    value_dict = request.form

    values=list(request.form.values())[0].split(',')
    # int_values=list(map(int,values))

    input_arr = np.zeros(len(columns))

    log10_columns = ['smoothness_mean','compactness_mean','fractal_dimension_mean','area_se',]
    log2_columns = ['radius_mean','texture_mean']
    log_columns = ['area_mean','radius_se','texture_se','perimeter_se','compactness_se','concave points_se','symmetry_se','radius_worst','perimeter_worst','area_worst','compactness_worst','fractal_dimension_se','fractal_dimension_worst']
    inverse_columns = ['perimeter_mean','symmetry_mean','smoothness_se','symmetry_worst']
    sqrt_columns = ['concavity_mean','concave points_mean','concavity_se','concavity_worst']


    for i in range(len(columns)):
        if columns[i] in log10_columns:
            input_arr[i] = np.log10(float(values[i]))
        elif columns[i] in log2_columns:
            input_arr[i] = np.log2(float(values[i]))
        elif columns[i] in log_columns:
            input_arr[i] = np.log(float(values[i]))
        elif columns[i] in inverse_columns:
            input_arr[i] = 1/(float(values[i]))
        elif columns[i] in sqrt_columns:
            input_arr[i] = np.sqrt(float(values[i]))
        else:
            input_arr[i] = float(values[i])

    scaled_input_arr = normal_scaler.transform([input_arr])   ## scaled_input_arr already became 2D after applying normal scaler         
    prediction = xg_model.predict(scaled_input_arr)               


    # print(input_arr)
    # print(prediction)

    if prediction == 0:
        outcome='B'
    else:
        outcome='M'

    return render_template('display.html',dict1 = value_dict, dict1_values = scaled_input_arr,result=outcome)


if __name__ == '__main__':
    app.run(debug=True)