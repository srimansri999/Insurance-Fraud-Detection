from src.exception.exception import CustomException
from src.logging.logger import logging
import os,sys
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.utils.main_utils.utils import load_numpy_array, evaluate_model, load_object, save_object
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from src.utils.ml_utils.model.estimator import InsuranceFraudModel

import mlflow
import mlflow.sklearn


class ModelTrainer:
    def __init__(self,model_trainer_config: ModelTrainerConfig,data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("Insurance Fraud Detection")

        except Exception as e:
            raise CustomException (e,sys)
    def track_mlflow(
        self,
        best_model,
        best_model_name,
        train_metric,
        test_metric
    ):
        try:
            print("Tracking URI:", mlflow.get_tracking_uri())
            with mlflow.start_run():

                mlflow.log_param("best_model", best_model_name)

                mlflow.log_metric("train_recall", train_metric.recall_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_f1", train_metric.f1_score)
                mlflow.log_metric("train_auc", train_metric.auc_roc_score)

                mlflow.log_metric("test_recall", test_metric.recall_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_f1", test_metric.f1_score)
                mlflow.log_metric("test_auc", test_metric.auc_roc_score)

                mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            raise CustomException(e, sys)

    
    def model_trainer(self, X_train,y_train, X_test,y_test):
        try:
            models = {
            "Logistic Regression": LogisticRegression(random_state=42,  max_iter=1000),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state =42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "XG Boost Classifier": XGBClassifier(eval_metric="logloss",random_state=42),
            "KNN Classifier": KNeighborsClassifier(),
            "Ada Boost Classifier": AdaBoostClassifier(random_state= 42),
            "Gradient Boost Classifier": GradientBoostingClassifier(random_state=42),
            "Cat Boost Classifier": CatBoostClassifier(verbose=False, random_state=42),
            "Support Vector Classifier": LinearSVC(random_state=42, max_iter=10000),
            "Naive Bayes Classifier": GaussianNB()
            }

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

            "Naive Bayes Classifier": {
                "var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
            }        }
            
            model_report, best_models = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
                scoring="recall"
            )

            best_model_name = max(
                model_report,
                key=lambda model: (
                    model_report[model]["recall"],
                    model_report[model]["auc"]
                )
            )

            best_model = best_models[best_model_name]

            logging.info(f"Best Model : {best_model_name}")
            logging.info(f"Recall : {model_report[best_model_name]['recall']:.4f}")
            logging.info(f"ROC-AUC : {model_report[best_model_name]['auc']:.4f}")
            



            y_train_pred = best_model.predict(X_train)

            if hasattr(best_model, "predict_proba"):
                y_train_prob = best_model.predict_proba(X_train)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_train_prob = best_model.decision_function(X_train)
            else:
                raise ValueError(
                    f"{best_model} does not support probability estimation."
                )



            classification_train_metric = get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred,
                y_pred_prob=y_train_prob
            )
            # Track the expirements with MLFLOW
            #self.track_mlflow(best_model, classification_train_metric)

            y_test_pred = best_model.predict(X_test)

            if hasattr(best_model, "predict_proba"):
                y_test_prob = best_model.predict_proba(X_test)[:, 1]
            elif hasattr(best_model, "decision_function"):
                y_test_prob = best_model.decision_function(X_test)
            else:
                raise ValueError(
                    f"{best_model} does not support probability estimation."
                )


            classification_test_metric = get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred,
                y_pred_prob=y_test_prob
            )

            #self.track_mlflow(best_model, classification_test_metric)
            # Calling At once
            self.track_mlflow(
                best_model=best_model,
                best_model_name=best_model_name,
                train_metric=classification_train_metric,
                test_metric=classification_test_metric
            )
  

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.model_trainer_trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)   

            insurance_fraud_model = InsuranceFraudModel(
                preprocessor=preprocessor,
                model=best_model
            )
            logging.info("Saving trained model...")
            save_object(self.model_trainer_config.model_trainer_trained_model_file_path, obj=insurance_fraud_model)
            os.makedirs("final_model", exist_ok=True)

            save_object(
                file_path="final_model/model.pkl",
                obj=best_model)
            
            logging.info("Model training completed successfully.")

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.model_trainer_trained_model_file_path,
                                train_metric_artifact=classification_train_metric,
                                test_metric_artifact=classification_test_metric)
            logging.info(f"Model Trainer artifact: {model_trainer_artifact}")   
            logging.info(f"Selected Model : {best_model_name}")
            logging.info(f"Training Recall : {classification_train_metric.recall_score:.4f}")
            logging.info(f"Testing Recall : {classification_test_metric.recall_score:.4f}")
            logging.info(f"Testing ROC-AUC : {classification_test_metric.auc_roc_score:.4f}")
  
            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException (e,sys)
        
    
    
    def initiate_model_trainer (self)-> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]
        
            model_trainer_artifact = self.model_trainer(X_train=x_train, y_train=y_train,
                                                       X_test=x_test, y_test=y_test)
            return model_trainer_artifact        



        except Exception as e:
            raise CustomException(e,sys)