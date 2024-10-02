import os,sys,dill
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging

def save_obj(obj, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)