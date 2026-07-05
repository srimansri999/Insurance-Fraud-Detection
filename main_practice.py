import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
import numpy as np
from src.logging.logger import logging
from src.exception.exception import CustomException
import os , sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV


def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        raise CustomException(e,sys)

def drop_unnecessary_cols(df: pd.DataFrame):
    try:
        columns_list = ['policy_number', 'insured_zip', 'incident_location']
        for col in columns_list:
            if col in df.columns.to_list():
                df.drop(col, inplace=True, axis = 1)
        return df
    except Exception as e:
        raise CustomException(e,sys)

def extracting_dates(df:pd.DataFrame):
    try:
        dates_list = ['policy_bind_date', 'incident_date', 'total_claim_amount']
        for col in dates_list:
            if col == 'policy_bind_date':
                df[col]=pd.to_datetime(df[col])
                df['policy_bind_year'] = df[col].dt.year
                df['policy_bind_month'] = df[col].dt.month
                df['policy_bind_day'] = df[col].dt.day
                df.drop(col, inplace=True, axis =1)
            elif col == 'incident_date':
                df[col]=pd.to_datetime(df[col])
                df['incident_month'] = df[col].dt.month
                df['incident_day'] = df[col].dt.day
                df.drop(col, inplace=True, axis=1)
            elif col == 'total_claim_amount':
                df.drop(col, inplace= True, axis = 1)
            else:
                pass
        return df
    except Exception as e:
        raise CustomException(e,sys)
    
def fill_na_and_question_mark(df:pd.DataFrame):
    try:
        df['collision_type'] = df['collision_type'].replace('?', 'unknown')
        df['property_damage'] = df['property_damage'].replace('?', 'unknown')
        df['police_report_available'] = df['police_report_available'].replace('?', 'unknown')
        df['authorities_contacted'] = df['authorities_contacted'].fillna('unknown')
        return df
    except Exception as e:
        raise CustomException(e,sys)

def fill_target_variable(df:pd.DataFrame):
    try:
        df['fraud_reported'] = df['fraud_reported'].map({"Y":1, "N":0})
        return df
    except Exception as e:
        raise CustomException(e,sys)

def dividing_data_into_train_and_test(df:pd.DataFrame):
    try:
        X = df.drop('fraud_reported', axis=1)
        y = df['fraud_reported']
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=42)
        return X_train,X_test,y_train,y_test
    except Exception as e:
        raise CustomException (e,sys)

def preprocessing_data(X_train, X_test):
    try:
        one_hot_encoding_columns = ['policy_state','policy_csl', 'insured_occupation' , 'insured_hobbies', 
                                    'insured_relationship','incident_type', 'collision_type', 'incident_severity','authorities_contacted', 
                                    'incident_state', 'incident_city', 'auto_make', 'auto_model', 'police_report_available',
                                    'property_damage']
        binary_encoding_columns = ['insured_sex', ]
        ordinal_encoding_columns = ['insured_education_level']
        numerical_scaling_columns = ['months_as_customer', 'age', 'policy_deductable','policy_annual_premium',
                                    'umbrella_limit', 'capital-gains','capital-loss', 'incident_hour_of_the_day', 
                                    'number_of_vehicles_involved','bodily_injuries','witnesses','injury_claim', 'property_claim',
                                    'vehicle_claim', 'auto_year', 'policy_bind_year', 'policy_bind_month',
                                        'policy_bind_day', 'incident_month','incident_day' ]
        education_order = [['High School','Associate' ,'College','Masters', 'JD', 'MD','PhD']]
        preprocessor = ColumnTransformer(transformers=[
            ('one-hot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False),one_hot_encoding_columns),
            ('binary', OrdinalEncoder(), binary_encoding_columns),
            ('ordinal',OrdinalEncoder(categories=education_order), ordinal_encoding_columns),
            ('numerical', StandardScaler(), numerical_scaling_columns)
                ], remainder='drop')
        categorical_columns = (one_hot_encoding_columns+binary_encoding_columns+ordinal_encoding_columns)
        X_train[categorical_columns] = X_train[categorical_columns].astype(str)
        X_test[categorical_columns] = X_test [categorical_columns].astype(str)
        
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)        
        os.makedirs('pickle_file', exist_ok=True)
        preprocessor_pickle_file_path = os.path.join('pickle_file', 'preprocessor.pkl')
        with open(preprocessor_pickle_file_path, 'wb') as file:
            pickle.dump(preprocessor, file)
        return X_train_preprocessed,X_test_preprocessed

        
    except Exception as e:
        raise CustomException(e,sys)

def model_trainer(X_train,X_test, y_train,y_test, models:dict):
    try:
        result = []
        for name, model in models.items():
            model = model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # ROC-AUC Scores
            if hasattr(model, "predict_proba"):
                y_train_score = model.predict_proba(X_train)[:, 1]
                y_test_score = model.predict_proba(X_test)[:, 1]

            elif hasattr(model, "decision_function"):
                y_train_score = model.decision_function(X_train)
                y_test_score = model.decision_function(X_test)

            else:
                y_train_score = None
                y_test_score = None

            result.append({
                "Model_Name": name,

                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),

                "train_precision": precision_score(y_train, y_train_pred),
                "test_precision": precision_score(y_test, y_test_pred),

                "train_recall": recall_score(y_train, y_train_pred),
                "test_recall": recall_score(y_test, y_test_pred),

                "train_f1_score": f1_score(y_train, y_train_pred),
                "test_f1_score": f1_score(y_test, y_test_pred),

                "train_roc_auc": (
                    roc_auc_score(y_train, y_train_score)
                    if y_train_score is not None else None
                ),

                "test_roc_auc": (
                    roc_auc_score(y_test, y_test_score)
                    if y_test_score is not None else None
                )
            })
        
        model_result = pd.DataFrame(result)        
        return  model_result    

    except Exception as e:
        raise CustomException(e,sys)

def model_names():
    try:
        models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree Classifier": DecisionTreeClassifier(),
        "Random Forest Classifier": RandomForestClassifier(),
        "XG Boost Classifier": XGBClassifier(),
        "KNN Classifier": KNeighborsClassifier(),
        "Ada Boost Classifier": AdaBoostClassifier(),
        "Gradient Boost Classifier": GradientBoostingClassifier(),
        "Cat Boost Classifier": CatBoostClassifier(),
        "Support Vector Classifier": LinearSVC(),
        "Navie Bayes Classifier": GaussianNB()}
        return models
    except Exception as e:
        raise CustomException(e,sys)


def hyperparameter_tuning(X_train, X_test, y_train, y_test,
                          models: dict,
                          params: dict,
                          scoring="recall"):
    try:

        results = []
        best_models= {}

        for name in models.keys():

            print(f"Tuning {name}...")

            random_search = RandomizedSearchCV(
                estimator=models[name],
                param_distributions=params[name],
                n_iter=20,
                cv=5,
                scoring=scoring,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )

            random_search.fit(X_train, y_train)

            best_model = random_search.best_estimator_

            best_models[name] = best_model

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # ROC-AUC
            if hasattr(best_model, "predict_proba"):
                train_score = best_model.predict_proba(X_train)[:, 1]
                test_score = best_model.predict_proba(X_test)[:, 1]

            elif hasattr(best_model, "decision_function"):
                train_score = best_model.decision_function(X_train)
                test_score = best_model.decision_function(X_test)

            else:
                train_score = None
                test_score = None

            results.append({

                "Model_Name": name,

                "Best_Params": random_search.best_params_,

                "train_accuracy": accuracy_score(y_train, y_train_pred),
                "test_accuracy": accuracy_score(y_test, y_test_pred),

                "train_precision": precision_score(y_train, y_train_pred),
                "test_precision": precision_score(y_test, y_test_pred),

                "train_recall": recall_score(y_train, y_train_pred),
                "test_recall": recall_score(y_test, y_test_pred),

                "train_f1_score": f1_score(y_train, y_train_pred),
                "test_f1_score": f1_score(y_test, y_test_pred),

                "train_roc_auc": roc_auc_score(y_train, train_score)
                if train_score is not None else None,

                "test_roc_auc": roc_auc_score(y_test, test_score)
                if test_score is not None else None

            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='test_recall', ascending= False)
        best_model_name = results_df.iloc[0]['Model_Name']
        best_model = best_models[best_model_name]        
        
        os.makedirs('pickle_file', exist_ok=True)
        model_pickle_file_path = os.path.join('pickle_file', 'model.pkl')
        with open(model_pickle_file_path, 'wb') as file:
            pickle.dump(best_model, file)

        return results_df
    except Exception as e:
        raise CustomException (e,sys)
    
def hyperparameter_params():
    try:
        params = {

            "Logistic Regression": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            },

            "Decision Tree Classifier": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [3, 5, 7, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8]
            },

            "Random Forest Classifier": {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            },

            "XG Boost Classifier": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7, 10],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
            },

            "KNN Classifier": {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"]
            },

            "Ada Boost Classifier": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
            },

            "Gradient Boost Classifier": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0]
            },

            "Cat Boost Classifier": {
                "iterations": [100, 200, 300],
                "depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1],
                "l2_leaf_reg": [1, 3, 5, 7]
            },

            "Support Vector Classifier": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "loss": ["hinge", "squared_hinge"]
            },

            "Navie Bayes Classifier": {
                "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
            }        }
        return params
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    logging.info('reading the dataset')
    df = read_excel(r'E:\Udemy\Project 4\Insurance Fraud Detection\Insurance_Data\Insurance Fraud Data.xlsx')
    logging.info('dropping the columns policy_number,insured_zip, incident_location ')
    df = drop_unnecessary_cols(df)
    logging.info('Creating Year, Month and Day for policy_bind_date and Month and Day for incident_date')
    df = extracting_dates(df)    
    logging.info('Filling the ? and nulls with unknown')
    df = fill_na_and_question_mark(df)
    logging.info('Filling Target Variable fraud_reported with Y:1, N:0')
    df = fill_target_variable(df)
    logging.info('Dividing data into train and test')
    X_train, X_test, y_train, y_test = dividing_data_into_train_and_test(df)
    logging.info(f"X_train has {X_train.shape[0]} rows and {X_train.shape[1]} columns")
    logging.info(f"X_test has {X_test.shape[0]} rows and {X_test.shape[1]} columns")
    logging.info(f"y_train has {y_train.shape[0]} rows")
    logging.info(f"y_test has {y_test.shape[0]} rows")
    X_train_preprocessed, X_test_preprocessed=preprocessing_data(X_train=X_train, X_test=X_test)
    logging.info(f"Preprocessed X_train and X_test data and saved the preprocessor pickle file")    
    logging.info('Started Model Training')
    models = model_names()
    result = model_trainer(X_train=X_train_preprocessed, X_test=X_test_preprocessed, y_train=y_train, y_test = y_test, models= models)
    result = result.sort_values(by='test_recall', ascending = False).reset_index (drop = True)    
    logging.info('Hyperparamter Tuning Started')
    params = hyperparameter_params()
    results_df = hyperparameter_tuning(X_train=X_train_preprocessed, X_test=X_test_preprocessed, y_train=y_train, y_test=y_test,
                          models= models,params= params,scoring="recall")
    print(result.head(10))
    print(results_df.head(10))