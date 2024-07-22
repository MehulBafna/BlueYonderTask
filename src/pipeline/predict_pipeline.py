import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
        season: int,
        year: int,
        hour: int,
        holiday: int,
        weekday: int,
        working_day: int,
        weather_situation: int,
        temperature: float,
        #feels_like_temperature: float,
        humidity: float,
        wind_speed:float,
        #casual: int,
        #registered: int,
        i_hour: int):

        self.season = season

        self.year = year

        self.hour = hour

        self.holiday = holiday

        self.weekday = weekday

        self.working_day = working_day

        self.weather_situation = weather_situation

        self.temperature = temperature

        #self.feels_like_temperature = feels_like_temperature 

        self.humidity = humidity 

        self.wind_speed = wind_speed 

        #self.casual = casual 

        #self.registered = registered 

        self.i_hour = i_hour



    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "season": [self.season],
                "year": [self.year],
                "hour": [self.hour],
                "holiday": [self.holiday],
                "weekday": [self.weekday],
                "working_day": [self.working_day],
                "weather_situation": [self.weather_situation],
                "temperature": [self.temperature],
                #"feels_like_temperature":[self.feels_like_temperature],
                "humidity": [self.humidity],
                "wind_speed": [self.wind_speed],
                #"casual": [self.casual],
                #"registered": [self.registered],
                "i_hour": [self.i_hour]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)