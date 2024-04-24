import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    
    def initiate_model_training(self, train_arr, test_arr):

        try:
            X_train =  train_arr[:, :-1]
            y_train = train_arr[:, -1]
            X_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            params = {
                'learning_rate': 0.2381413287859572, 
                'n_estimators': 439, 
                'max_depth': 6, 
                'min_child_weight': 3, 
                'gamma': 0.3308509875421135, 
                'subsample': 0.9642190731549022, 
                'colsample_bytree': 0.40885708517990055, 
                'reg_alpha': 0.35435676335715693,
                'random_state' : 102
            }

            logging.info("Model training started")

            model = XGBClassifier()
            model.set_params(**params)
            model.fit(X_train, y_train)

            logging.info("Model training completed")

            save_object(file_path = self.model_trainer_config.model_obj_file_path,
                        obj= model)
            
            logging.info("Model saved")



            logging.info("Model evaluation started")

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logging.info("Model evaluation completed")

            return accuracy


        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


