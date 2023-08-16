from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form.get('age'),
            sex=int(request.form.get('sex')),
            cp=int(request.form.get('cp')),
            trestbps=request.form.get('trestbps'),
            chol=request.form.get('chol'),
            fbs=int(request.form.get('fbs')),
            restecg=int(request.form.get('restecg')),
            thalach=request.form.get('thalach'),
            exang=int(request.form.get('exang')),
            oldpeak=request.form.get('oldpeak'),
            slope=int(request.form.get('slope')),
            ca=request.form.get('ca'),
            thal=int(request.form.get('thal'))
            
        )
        
        pred_df = data.get_data_as_dataframe()
        
        predict_pipeline = PredictPipeline()
        results,probability=predict_pipeline.predict(pred_df)
        
        text = f"you have {(round(probability[0][1],2))*100}% of chance of having a heart disease."
        
        if results == 1:
            text += " Its advisable to get a checkup"
        else :
            text += " You are healthy"
        
        return render_template('home.html', text=text)

if __name__ == '__main__':
    app.run(debug=True)