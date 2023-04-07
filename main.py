from flask import Flask,request,render_template,url_for
<<<<<<< HEAD
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
=======
import pickle as pkl

import numpy as np
import pandas as pd
import time

from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,LancasterStemmer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost

from sklearn.metrics import accuracy_score
>>>>>>> refs/remotes/origin/master


app = Flask('__main__')

<<<<<<< HEAD
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
=======
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\svm_model.pkl','rb') as file1:
    svm_model=pkl.load(file1)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\xgb_model.pkl','rb') as file2:
    xgb_model=pkl.load(file2)
    
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\rfc_model.pkl','rb') as file3:
    rfc_model=pkl.load(file3)
    
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\mnb_model.pkl','rb') as file4:
    mnb_model=pkl.load(file4)

with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\tfidf.pkl','rb') as file5:
    tfidf=pkl.load(file5)
    
with open(r'C:\Users\Omkar\Desktop\Practice\Datasets\Twitter Sentiment Analysis\Artifacts\cv.pkl','rb') as file6:
    cv=pkl.load(file6)


>>>>>>> refs/remotes/origin/master

@app.route('/')
def render():
    return render_template('index.html')

@app.route('/input', methods=['GET','POST'])
def predict():
<<<<<<< HEAD
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
    print(values)
=======
    
    input_dict = request.form
    tweet=list(input_dict.values())[0]
    inverted_comma="'"

    if  inverted_comma in tweet:
        modified_tweet=tweet.replace(inverted_comma,'')
    else:
        modified_tweet=tweet
      
        
    def tokenization():
      words=word_tokenize(modified_tweet)
      return words
    tokens=tokenization()


    def cleaning():
      clean_text=[i for i in tokens if i not in punctuation]
      return clean_text
    clean_text=cleaning()
    

    def normalize():
      normal_text=[i.lower() for i in clean_text]
      return normal_text
    normal_text=normalize()
    

    stop=stopwords.words('english')
    def stop_removal():
      stop_text=[i for i in normal_text if i not in stop]
      return stop_text
    stop_text=stop_removal()
    

    lemma=WordNetLemmatizer()
    def lemmatization():
      l1=[]
      for i in stop_text:
          word=lemma.lemmatize(i)
          l1.append(word)
      return l1
    l1=lemmatization()


    def string():
      strings=' '.join(l1)
      return strings
    final_tweet=string()

    matrix=tfidf.transform([final_tweet]).A
    prediction=svm_model.predict(matrix)

    if prediction == 0:
       tweet_nature = 'Irrelevent'
    if prediction == 1:
       tweet_nature = 'Negative'
    if prediction == 2:
       tweet_nature = 'Neutral'
    if prediction == 3:
       tweet_nature = 'Positive'
       
    
    return render_template('display.html',dict1_values = modified_tweet, output=tweet_nature)
>>>>>>> refs/remotes/origin/master

if __name__ == '__main__':
    app.run(debug=True)