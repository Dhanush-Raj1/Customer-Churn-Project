import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import Predict, NewData

application = Flask(__name__)
app = application


# route for home page
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_newdata():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = NewData( gender = request.form.get('gender'),
                        SeniorCitizen = request.form.get('SeniorCitizen'), 
                        Partner = request.form.get('Partner'), 
                        Dependents = request.form.get('Dependents'), 
                        tenure = float(request.form.get('tenure')), 
                        PhoneService = request.form.get('PhoneService'), 
                        MultipleLines = request.form.get('MultipleLines'), 
                        InternetService = request.form.get('InternetService'), 
                        OnlineSecurity = request.form.get('OnlineSecurity'), 
                        OnlineBackup = request.form.get('OnlineBackup'), 
                        DeviceProtection = request.form.get('DeviceProtection'), 
                        TechSupport = request.form.get('TechSupport'), 
                        StreamingTV = request.form.get('StreamingTV'), 
                        StreamingMovies = request.form.get('StreamingMovies'), 
                        Contract = request.form.get('Contact'), 
                        PaperlessBilling = request.form.get('PaperlessBilling'), 
                        PaymentMethod = request.form.get('PaymentMethod'), 
                        MonthlyCharges = float(request.form.get('MonthlyCharges')), 
                        TotalCharges = float(request.form.get('TotalCharges'))   )
        
        newdata_df = data.get_data_as_dataframe()
        print(newdata_df)
        
        prediction = Predict()
        results = prediction.predict_data(newdata_df)
        
        return render_template('home.html', results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)
    
        
        