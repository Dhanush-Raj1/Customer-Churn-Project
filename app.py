import pandas as pd
import numpy as np
from src.logger import logging
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import Predict, NewData

application = Flask(__name__)
app = application


# route for home page   
@app.route('/')
def home():
    return render_template('home_page.html')


# route for predict page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    
    if request.method == 'GET':
        return render_template('predict_page.html')
    
    
    else:
        try:
            logging.info("POST Request has been made.")
            data = NewData( gender = request.form.get('Gender'),
                            SeniorCitizen = request.form.get('Senior Citizen'), 
                            Partner = request.form.get('Partner'), 
                            Dependents = request.form.get('Dependents'), 
                            tenure = float(request.form.get('Tenure (months)')), 
                            PhoneService = request.form.get('Phone Service'), 
                            MultipleLines = request.form.get('Multiple Lines'), 
                            InternetService = request.form.get('Internet Service'), 
                            OnlineSecurity = request.form.get('Online Security'), 
                            OnlineBackup = request.form.get('Online Backup'), 
                            DeviceProtection = request.form.get('Device Protection'), 
                            TechSupport = request.form.get('Tech Support'), 
                            StreamingTV = request.form.get('Streaming TV'), 
                            StreamingMovies = request.form.get('Streaming Movies'), 
                            Contract = request.form.get('Contract Type'), 
                            PaperlessBilling = request.form.get('Paperless Billing'), 
                            PaymentMethod = request.form.get('Payment Method'), 
                            MonthlyCharges = float(request.form.get('Monthly Charges', 0)), 
                            TotalCharges = float(request.form.get('Total Charges', 0))   )
            
            df = data.get_data_as_dataframe()
            logging.info("Data has been converted to a DataFrame")
            print(df)
            
            prediction = Predict()
            logging.info("Predicting process has been initialized.")
            results = prediction.predict_data(df)
            print("Prediction results:", results)
            
            return render_template('predict_page.html', results=results)
        
        except ValueError as e:
            return render_template('predict_page.html', error=str(e))
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
    
        
        