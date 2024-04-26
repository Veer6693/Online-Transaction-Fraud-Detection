import sys
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,data):
        try:

            preprocessor_obj = load_object('artifacts/preprocessor.pkl')
            model_obj = load_object('artifacts/model.pkl')

            logging.info("preprocessing started")
            data = preprocessor_obj.transform(data)
            logging.info("preprocessing completed")

            logging.info("prediction started")
            prediction = model_obj.predict(data)
            logging.info("prediction done")
            return prediction

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


