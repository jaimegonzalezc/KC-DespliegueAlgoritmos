from funciones import args,load_dataset,preprocess_text_nltk,vectorizer,model_train,log_model_metrics

def main():
    args_values = args()
    df_train, df_test = load_dataset(args_values.categories)
    df_train['clean_text'] = df_train['text'].apply(preprocess_text_nltk) 
    df_test['clean_text'] = df_test['text'].apply(preprocess_text_nltk)
    X_train, X_test, y_train, y_test = vectorizer(df_train, df_test)
    modelo = model_train(X_train, y_train, args_values.max_iter)
    log_model_metrics(modelo, X_test, y_test)

    if __name__ == '__main__':
        modelo()

