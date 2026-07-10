
import os

PIPELINE_NAME: str = 'InsuranceFraudDetection'
ARTIFACT_DIR:str = 'Artifact'

TRAIN_FILE_NAME: str = 'train.csv'
TEST_FILE_NAME: str = 'test.csv'
FILE_NAME: str ='insurancefrauddetection.csv'
TARGET_COLUMN : str = 'fraud_reported'

RAW_SCHEMA_FILE_PATH = os.path.join('data_schema', 'raw_schema.yaml')
PROCESSED_SCHEMA_FILE_PATH = os.path.join('data_schema', 'preprocessed_schema.yaml')


'''
Data Ingestion Constants - All the variable names start with DATA_INGESTION
'''

DATA_INGESTION_DIR_NAME: str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_FILE_PATH:str = 'feature_store'
DATA_INGESTION_INGESTED_DATA: str = 'ingested'
DATA_INGESTIOON_TRAIN_TEST_SPLIT_RATION:float = 0.2
DATA_INGESTION_FILE_PATH:str  = r'E:\Udemy\Project 4\Insurance Fraud Detection\Insurance_Data\Insurance Fraud Data.xlsx'

'''
Data Validation Constants - All the variable names starts with DATA_VALIDATION
'''
DATA_VALIDATION_DIR_NAME: str = 'data_validation'
DATA_VALIDATION_VALID_DIR_NAME: str = 'validated_data'
DATA_VALIDATION_INVALID_DIR_NAME: str = 'invalid_data'
DATA_VALIDATION_DIRFT_REPORT_FILE_PATH: str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "raw_data_report.yaml"

'''
Data Transformation Constants - All the variables names starts with DATA_TRANSFORMATION
'''

DATA_TRANSFORMATION_DIR_NAME: str = 'data_transformation'
DATA_TRANSFORMATION_VALID_DIR_NAME: str = 'validated_data'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = 'transformed'
DATA_TRANSFORMATION_INVALID_DIR_NAME: str = 'invalid_data'
DATA_TRANSFORMATION_DRIFT_REPORT_FILE_PATH: str = 'drift_report'
DATA_TRANSFORMATION_DRIFT_REPORT_FILE_NAME: str = 'preprocessed_data_report.yaml'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = 'transformed_object'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME: str = 'preprocessing.pkl'

DATA_TRANSFORMATION_DROPPING_COLUMNS: list = ['total_claim_amount', 'policy_number', 'insured_zip', 
                                              'incident_location','policy_bind_date','incident_date']

DATA_TRANSFORMATION_IMPUTATION_COLUMNS: list = ['collision_type', 'property_damage', 'police_report_available', 'authorities_contacted']

DATA_TRANSFORMATION_ONE_HOT_ENCODING_COLUMNS: list = ['policy_state','policy_csl', 'insured_occupation' , 'insured_hobbies', 
                                    'insured_relationship','incident_type', 'collision_type', 'incident_severity','authorities_contacted', 
                                    'incident_state', 'incident_city', 'auto_make', 'auto_model', 'police_report_available',
                                    'property_damage']
DATA_TRANSFORMATION_BINARAY_ENCODING_COLUMNS: list = ['insured_sex', ]
DATA_TRANSFORMATION_ORDINAL_ENCODING_COLUMNS: list = ['insured_education_level']
DATA_TRANSFORMATION_NUMERICAL_SCALING_COLUMNS: list = ['months_as_customer', 'age', 'policy_deductable','policy_annual_premium',
                                    'umbrella_limit', 'capital-gains','capital-loss', 'incident_hour_of_the_day', 
                                    'number_of_vehicles_involved','bodily_injuries','witnesses','injury_claim', 'property_claim',
                                    'vehicle_claim', 'auto_year', 'policy_bind_year', 'policy_bind_month',
                                        'policy_bind_day', 'incident_month','incident_day' ]
DATA_TRANSFORMATION_EDUCATION_ORDER = [['High School','Associate' ,'College','Masters', 'JD', 'MD','PhD']]


DATA_TRANSAFORMATION_CATEGORICAL_COLUMNS = (DATA_TRANSFORMATION_ONE_HOT_ENCODING_COLUMNS + DATA_TRANSFORMATION_BINARAY_ENCODING_COLUMNS +
    DATA_TRANSFORMATION_ORDINAL_ENCODING_COLUMNS)




