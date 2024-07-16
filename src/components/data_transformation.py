import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

import warnings
import os 

# Set LOKY_MAX_CPU_COUNT to the number of cores you want to use
os.environ["LOKY_MAX_CPU_COUNT"] = "4" 

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not find the number of physical cores.*")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        logging.info(11)
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ["temperature","feels_like_temperature","registered","casual","wind_speed","humidity",'hour']
            categorical_columns = [
                "season","year","weekday","working_day","weather_situation","holiday",'i_hour'
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median"))
                #("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder())
                #("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(r"C:\\Users\\Mehul\BlueYonderTask\\artifacts\\train.csv")
            test_df=pd.read_csv(r"C:\\Users\\Mehul\\BlueYonderTask\\artifacts\\test.csv")

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            

            # Rename columns
            
            target_column_name="count"

            Q1 = train_df['count'].quantile(0.25)
            Q3 = train_df['count'].quantile(0.75)
            IQR = Q3 - Q1

            # Define bounds for the outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify the outliers
            train_df = train_df[(train_df['count'] >= lower_bound) & (train_df['count'] <= upper_bound)]
            train_df = train_df.reset_index(drop=True)
            train_df.loc[:,['temperature']] = train_df.loc[:,['temperature']].apply(lambda x: 47*x -8)
            train_df.loc[:,['feels_like_temperature']] = train_df.loc[:,['feels_like_temperature']].apply(lambda x: 66*x -16)
            train_df.loc[:,['humidity']]=train_df.loc[:,['humidity']]*67
            train_df.loc[:,['wind_speed']]= train_df.loc[:,['wind_speed']]*100
            #train_df['i_hour'] = train_df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 20) else 0)
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            test_df = test_df[(test_df['count'] >= lower_bound) & (test_df['count'] <= upper_bound)]
            test_df = test_df.reset_index(drop=True)
            test_df.loc[:,['temperature']] = test_df.loc[:,['temperature']].apply(lambda x: 47*x -8)
            test_df.loc[:,['feels_like_temperature']] = test_df.loc[:,['feels_like_temperature']].apply(lambda x: 66*x -16)
            test_df.loc[:,['humidity']]=test_df.loc[:,['humidity']]*67
            test_df.loc[:,['wind_speed']]= test_df.loc[:,['wind_speed']]*100
            #test_df['i_hour'] = test_df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (16 <= x <= 20) else 0)
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            preprocessing_obj=self.get_data_transformer_object()
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
