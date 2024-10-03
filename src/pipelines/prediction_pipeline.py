import sys,os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_obj

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, X):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_obj(model_path)
            preprocessor = load_obj(preprocessor_path)
            data = preprocessor.transform(X)
            pred = model.predict(data)
            return pred
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                Income:int,
                Age:int,
                CURRENT_HOUSE_YRS:int,
                Experience:int,
                House_Ownership:str, 
                Car_Ownership:str, 
                Married_Single:str):
        self.Income = Income
        self.Age = Age
        self.CURRENT_HOUSE_YRS = CURRENT_HOUSE_YRS
        self.Experience = Experience
        self.House_Ownership = str.lower(House_Ownership)
        self.Car_Ownership = str.lower(Car_Ownership)
        self.Married_Single = str.lower(Married_Single)
    
    def data_as_data_frame(self):    
        return pd.DataFrame({
            "Income": [self.Income],
            "Age": [self.Age],
            "CURRENT_HOUSE_YRS": [self.CURRENT_HOUSE_YRS],
            "Experience": [self.Experience],
            "House_Ownership": [self.House_Ownership],
            "Car_Ownership": [self.Car_Ownership],
            "Married/Single": [self.Married_Single]
        })
        