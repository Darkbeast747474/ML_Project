import os, sys, pandas
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestion:
    train_data_path: str = os.path.join("artifacts/Data", "train.csv")
    test_data_path: str = os.path.join("artifacts/Data", "test.csv")
    raw_data_path: str = os.path.join("artifacts/Data", "data.csv")

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Initiated")
        try:
            df = pandas.read_csv("Notebooks/Training Data.csv")
            logging.info("Dataset Fetched as Pandas Dataframe")

            os.makedirs(os.path.dirname(self.train_data_path),exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)

            logging.info("Train-Test Split Initiated")
            train, test = train_test_split(df, test_size=0.3, random_state=42)

            train.to_csv(self.train_data_path, index=False, header=True)
            test.to_csv(self.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Successfully Terminated")

            return (
                self.train_data_path,
                self.test_data_path,
            )
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
                

