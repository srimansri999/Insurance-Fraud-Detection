from src.exception.exception import CustomException
from src.logging.logger import logging
import sys, os
import pandas as pd
from src.utils.ml_utils.model.estimator import InsuranceFraudModel
from src.utils.main_utils.utils import load_object
import traceback
from pathlib import Path

# Absolute path to project root, independent of where you launch python from
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust if needed


class PredictPipeline:
    def __init__(self):
        pass

    def predict (self, features):
        try:
            preprocessor_path = os.path.join("final_model", "preprocessor.pkl")
            model_path = os.path.join("final_model", "model.pkl")
          

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
           

      
            print(preprocessor.feature_names_in_)

            x_transform = preprocessor.transform(features)
           

            insurance_fraud_model = InsuranceFraudModel( model = model)
            probability_prediction = insurance_fraud_model.predict_proba(x_transform)
            prediction = insurance_fraud_model.predict(x_transform)
    

          

            return {
                "prediction": prediction,
                "probability": probability_prediction
            }


        except Exception as e:
            
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
            months_as_customer: float,
            age: float ,
            policy_state: str,
            policy_csl: str,
            policy_deductable : float,
            policy_annual_premium : float,
            umbrella_limit :float,
            insured_sex : str, 
            insured_education_level : str,
            insured_relationship :str ,
            insured_occupation: str, 
            insured_hobbies: str,  
            capital_gains: float,
            capital_loss : float, 
            incident_type :str, 
            collision_type : str, 
            incident_severity : str, 
            authorities_contacted : str,
            incident_state : str,
            incident_city : str, 
            incident_hour_of_the_day : float, 
            number_of_vehicles_involved : float, 
            bodily_injuries : float, 
            witnesses : float, 
            property_damage: str, 
            police_report_available : str, 
            auto_make : str, 
            auto_model : str, 
            auto_year : float ,
            injury_claim : float,
            property_claim : float,
            vehicle_claim :  float, 
            policy_bind_year : float, 
            policy_bind_month : float, 
            policy_bind_day : float, 
            incident_month : float, 
            incident_day : float                  
                 ):
        self.months_as_customer =   months_as_customer
        self.age = age
        self.policy_state = policy_state
        self.policy_csl = policy_csl
        self.policy_deductable = policy_deductable
        self.policy_annual_premium =policy_annual_premium 
        self.umbrella_limit =umbrella_limit
        self.insured_sex = insured_sex
        self.insured_education_level = insured_education_level
        self.insured_relationship = insured_relationship
        self.insured_occupation= insured_occupation
        self.insured_hobbies = insured_hobbies
        self.capital_gains = capital_gains
        self.capital_loss = capital_loss
        self.incident_type = incident_type
        self.collision_type = collision_type
        self.incident_severity = incident_severity
        self.authorities_contacted = authorities_contacted
        self.incident_state = incident_state
        self.incident_city = incident_city
        self.incident_hour_of_the_day = incident_hour_of_the_day 
        self.number_of_vehicles_involved = number_of_vehicles_involved
        self.bodily_injuries = bodily_injuries
        self.witnesses = witnesses
        self.property_damage= property_damage
        self.police_report_available = police_report_available
        self.auto_make = auto_make
        self.auto_model = auto_model
        self.auto_year = auto_year
        self.injury_claim = injury_claim
        self.property_claim = property_claim
        self.vehicle_claim = vehicle_claim
        self.policy_bind_year = policy_bind_year 
        self.policy_bind_month = policy_bind_month 
        self.policy_bind_day = policy_bind_day
        self.incident_month = incident_month
        self.incident_day = incident_day  
    
    def get_data_as_dataframe (self):
        try:
            custom_data_input_dict = {
                "months_as_customer": [self.months_as_customer],
                "age": [self.age],
                "policy_state": [self.policy_state],
                "policy_csl": [self.policy_csl],
                "policy_deductable": [self.policy_deductable],
                "policy_annual_premium": [self.policy_annual_premium],
                "umbrella_limit": [self.umbrella_limit],
                "insured_sex": [self.insured_sex],
                "insured_education_level": [self.insured_education_level],
                "insured_relationship": [self.insured_relationship],
                "insured_occupation": [self.insured_occupation],
                "insured_hobbies": [self.insured_hobbies],
                "capital-gains": [self.capital_gains],
                "capital-loss": [self.capital_loss],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "authorities_contacted": [self.authorities_contacted],
                "incident_state": [self.incident_state],
                "incident_city": [self.incident_city],
                "incident_hour_of_the_day": [self.incident_hour_of_the_day],
                "number_of_vehicles_involved": [self.number_of_vehicles_involved],
                "bodily_injuries": [self.bodily_injuries],
                "witnesses": [self.witnesses],
                "property_damage": [self.property_damage],
                "police_report_available": [self.police_report_available],
                "auto_make": [self.auto_make],
                "auto_model": [self.auto_model],
                "auto_year": [self.auto_year],
                "injury_claim": [self.injury_claim],
                "property_claim": [self.property_claim],
                "vehicle_claim": [self.vehicle_claim],
                "policy_bind_year": [self.policy_bind_year],
                "policy_bind_month": [self.policy_bind_month],
                "policy_bind_day": [self.policy_bind_day],
                "incident_month": [self.incident_month],
                "incident_day": [self.incident_day],
            }
            df =  pd.DataFrame(custom_data_input_dict)
            
            return df
        except Exception as e:
            
            raise CustomException(e,sys)

    
        
