from flask import Flask, render_template, request
import sys, os
import traceback
from src.exception.exception import CustomException
from src.pipeline.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application



@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    try:
        if request.method  == 'GET':
            return render_template('home.html')
        else:          

          
            data = CustomData(
                months_as_customer=float(request.form.get('months_as_customer')),
                age=float(request.form.get('age')),
                policy_state=request.form.get('policy_state'),
                policy_csl=request.form.get('policy_csl'),
                policy_deductable=float(request.form.get('policy_deductable')),
                policy_annual_premium=float(request.form.get('policy_annual_premium')),
                umbrella_limit=float(request.form.get('umbrella_limit')),
                insured_sex=request.form.get('insured_sex'),
                insured_education_level=request.form.get('insured_education_level'),
                insured_relationship=request.form.get('insured_relationship'),
                insured_occupation=request.form.get('insured_occupation'),
                insured_hobbies=request.form.get('insured_hobbies'),
                capital_gains=float(request.form.get('capital_gains')),
                capital_loss=float(request.form.get('capital_loss')),
                incident_type=request.form.get('incident_type'),
                collision_type=request.form.get('collision_type'),
                incident_severity=request.form.get('incident_severity'),
                authorities_contacted=request.form.get('authorities_contacted'),
                incident_state=request.form.get('incident_state'),
                incident_city=request.form.get('incident_city'),
                incident_hour_of_the_day=float(request.form.get('incident_hour_of_the_day')),
                number_of_vehicles_involved=float(request.form.get('number_of_vehicles_involved')),
                bodily_injuries=float(request.form.get('bodily_injuries')),
                witnesses=float(request.form.get('witnesses')),
                property_damage=request.form.get('property_damage'),
                police_report_available=request.form.get('police_report_available'),
                auto_make=request.form.get('auto_make'),
                auto_model=request.form.get('auto_model'),
                auto_year=float(request.form.get('auto_year')),
                injury_claim=float(request.form.get('injury_claim')),
                property_claim=float(request.form.get('property_claim')),
                vehicle_claim=float(request.form.get('vehicle_claim')),
                policy_bind_year=float(request.form.get('policy_bind_year')),
                policy_bind_month=float(request.form.get('policy_bind_month')),
                policy_bind_day=float(request.form.get('policy_bind_day')),
                incident_month=float(request.form.get('incident_month')),
                incident_day=float(request.form.get('incident_day'))
            )
            
            new_data_dataframe = data.get_data_as_dataframe()
                    
            

    

            prediction_pipeline = PredictPipeline()
            prediction_result = prediction_pipeline.predict(new_data_dataframe)

            

            prediction = prediction_result["prediction"]
            probability_prediction = prediction_result["probability"]

            predicted_class = int(prediction[0])

            results = "Fraud Detected" if predicted_class == 1 else "Genuine Claim"

            confidence = round(
                probability_prediction[0][predicted_class] * 100,
                2
            )
            #print(f"[DEBUG] Final result: {results}, confidence: {confidence}")
            return render_template("home.html", results=results,confidence=confidence)


    except Exception as e:
        #print("[ERROR] Exception in /predictdata route")
        
        raise CustomException(e,sys)
     



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)