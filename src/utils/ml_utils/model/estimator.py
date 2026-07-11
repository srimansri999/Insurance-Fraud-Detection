from src.exception.exception import CustomException
from src.logging.logger import logging

import sys

class InsuranceFraudModel:
    def __init__(self,preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict_proba(self, x):
        try:
            x_transform = self.preprocessor.transform(x)

            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(x_transform)

            return self.model.decision_function(x_transform)

        except Exception as e:
            raise CustomException(e, sys)
