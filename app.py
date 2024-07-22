import pickle 
from flask import Flask, request,render_template
import numpy as np 
import pandas as pd 
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import gunicorn

application=Flask(__name__)

app = application 

## Route for a Home Page 

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            season=int(request.form.get('season')),
            year=int(request.form.get('year')),
            hour=int(request.form.get('hour')),
            holiday=int(request.form.get('holiday')),
            weekday=int(request.form.get('weekday')),
            working_day=int(request.form.get('working_day')),
            weather_situation=int(request.form.get('weather_situation')),
            temperature=float(request.form.get('temperature')),
            #feels_like_temperature=float(request.form.get('feels_like_temperature')),
            humidity=float(request.form.get('humidity')),
            wind_speed=float(request.form.get('wind_speed')),
            #casual=int(request.form.get('casual')),
            #registered=int(request.form.get('registered')),
            i_hour=int(request.form.get('i_hour'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=int(results[0]))
    
if __name__=="__main__":
    app.run(host="0.0.0.0")