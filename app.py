from src.logger import logging
from src.pipelines.predict_pipeline import Predict, NewData

from flask import Flask, request, render_template



application = Flask(__name__)
app = application


# route for home page   
@app.route('/')
def home():
    return render_template('home_page.html')

 
# route for predict page
@app.route('/predictdata', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict_page.html')
    
    else:
        logging.info("POST Request has been made.")
        data = NewData( gender = request.form.get('Gender'),
                        SeniorCitizen = request.form.get('Senior Citizen'), 
                        Partner = request.form.get('Partner'), 
                        Dependents = request.form.get('Dependents'), 
                        tenure = request.form.get('Tenure (months)'), 
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
                        MonthlyCharges = float(request.form.get('Monthly Charges')), 
                        TotalCharges = float(request.form.get('Total Charges'))  )
                
        df = data.get_data_as_dataframe()
        logging.info("Data has been converted in to a DataFrame.")
        logging.info(f"DataFrame: \n{df.head()}")
        print(df.head())    
                
        prediction = Predict()
        result = prediction.predict_data(df)
        print("Prediction result:", result)
        logging.info("Prediction Result: {result}")
        
        result_text = ""
        if result == 0:
            result_text = "Your customer will not churn from your business."
        else:
            result_text = "Your customer will churn from your business."
                 
        return render_template('predict_page.html', result=result[0], result_text = result_text)
        
        #except Exception as e:
           # return render_template('predict_page.html', error=str(e))
        
    #else: 
        #return render_template('predict_page.html')
        
           
    

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False) #host="0.0.0.0"
    
        
        