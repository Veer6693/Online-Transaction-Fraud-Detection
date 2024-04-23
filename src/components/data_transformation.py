import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    
    def clean_data(self, df) -> pd.DataFrame:
      

        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
        

        df['Transaction Day'] = df["Transaction Date"].dt.day
        df["Transaction DOW"] = df["Transaction Date"].dt.day_of_week
        df["Transaction Month"] = df["Transaction Date"].dt.month
        

        mean_value = np.round(df['Customer Age'].mean(),0) 
        df['Customer Age'] = np.where(df['Customer Age'] <= -9, 
                                        np.abs(df['Customer Age']), 
                                        df['Customer Age'])

        df['Customer Age'] = np.where(df['Customer Age'] < 9, 
                                        mean_value, 
                                        df['Customer Age'])
        
    
        df["Is Address Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)
        

        df.drop(columns=["Transaction ID", "Customer ID", "Customer Location",
                        "IP Address", "Transaction Date","Shipping Address","Billing Address"], inplace=True)
        
        
        int_col = df.select_dtypes(include="int").columns
        float_col = df.select_dtypes(include="float").columns
        
        df[int_col] = df[int_col].apply(pd.to_numeric, downcast='integer')
        df[float_col] = df[float_col].apply(pd.to_numeric, downcast='float')
        
        return df

    def get_data_transformer_object(self):
        try:
            numeric_col = ['Transaction Amount', 'Quantity', 'Customer Age',  'Account Age Days','Transaction Hour',
                           'Transaction Day', 'Transaction DOW', 'Transaction Month' ]
            logging.info(f"numerical column : {numeric_col}")

            cat_col = ['Payment Method', 'Product Category', 'Device Used']
            logging.info(f"categorical column : {cat_col}")


            num_pipeline = Pipeline(
                steps=[
                    ('scaler', StandardScaler()),
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('encoder', OneHotEncoder()),
                ]
            )


            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numeric_col),
                    ('cat_pipeline', cat_pipeline, cat_col),
                ],
                remainder='passthrough'
            )

            return preprocessor


        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
        
    
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read training data and test data")

            logging.info("Data Cleaning started for training and testing data...")
            clean_train_df = self.clean_data(train_df)
            clean_test_df = self.clean_data(test_df)

            logging.info("Data Cleaning completed for training and testing data...")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            target_col = 'Is Fraudulent'

            input_train_df = clean_train_df.drop(columns=[target_col])
            target_train_df = clean_train_df[target_col]

            input_test_df = clean_test_df.drop(columns=[target_col])
            target_test_df = clean_test_df[target_col]


            logging.info("Data Transformation started for training and testing data...")

            transformed_train_arr = preprocessor_obj.fit_transform(input_train_df)
            transformed_test_arr = preprocessor_obj.transform(input_test_df)

            train_arr = np.c_[transformed_train_arr, np.array(target_train_df)]

            test_arr = np.c_[transformed_test_arr, np.array(target_test_df)]

            logging.info("Data Transformation completed for training and testing data")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)