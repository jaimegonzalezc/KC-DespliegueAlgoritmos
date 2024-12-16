import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import nltk
import re
import argparse

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

nltk.download('punkt_tab')
nltk.download('stopwords')

def args():
    parser = argparse.ArgumentParser(description='__main__ de la aplicación con argumentos de entrada.')
    parser.add_argument('--max_iter', type=int, help='Máximas iteraciones del modelo')
    parser.add_argument('--categories', nargs='+', type=str, help='Lista de categorías')    
    return parser.parse_args()

def load_dataset(categories):
    #categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    df_train = pd.DataFrame({'text': newsgroups_train.data, 'target': newsgroups_train.target}) 
    df_test = pd.DataFrame({'text': newsgroups_test.data, 'target': newsgroups_test.target})
    return df_train, df_test

def preprocess_text_nltk(text):
    text = re.sub(r'From:.*\n', '', text)
    text = re.sub(r'Subject:.*\n', '', text)
    text = re.sub(r'Lines:.*\n', '', text)
    text = re.sub(r'To:.*\n', '', text)
    text = re.sub(r'Contact:.*\n', '', text)
    text = re.sub(r'NOTE.*\n', '', text)
    text = re.sub(r'\n---+\n', '', text)
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english')) 
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def vectorizer(df_train, df_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(df_train['clean_text'])
    X_test = vectorizer.transform(df_test['clean_text'])
    y_train = df_train['target']
    y_test = df_test['target']
    return X_train, X_test, y_train, y_test

def model_train(X_train, y_train, max_iter):
    model = LogisticRegression(max_iter)
    model.fit(X_train, y_train)
    return model

def log_model_metrics(model, X_test, y_test):
    # Evaluar el modelo y registrar las métricas
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')

    with mlflow.start_run():
        # Registrar parámetros y métricas
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("vectorizer", "TfidfVectorizer")
        mlflow.log_param("max_iter", 1000)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Registrar el modelo entrenado
        mlflow.sklearn.log_model(model, "model")

