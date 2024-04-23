import os 
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion')
        try:
            train_df = pd.read_csv('notebook\\data\\Fraudulent_E-Commerce_Transaction_Data.csv')
            test_df = pd.read_csv('notebook\\data\\Fraudulent_E-Commerce_Transaction_Data_2.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data ingestion completed successfully')


            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                )

                
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
            
if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()


    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    print(train_arr.shape, test_arr.shape)

