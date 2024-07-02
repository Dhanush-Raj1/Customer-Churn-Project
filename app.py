import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application


# route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=='GET':
        return render_template('home.html')
    else:
        pass 