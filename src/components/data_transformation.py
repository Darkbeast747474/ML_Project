import os, sys, pandas, pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestion
from src.utils import save_obj
from src.components.model_tranier import ModelTrainer


@dataclass
class DataTransformation:
    train_data_path: str = DataIngestion.train_data_path
    test_data_path: str = DataIngestion.test_data_path
    preprocessed_obj_path: str = os.path.join("artifacts", "preprocessor.pkl")

    def initiate_data_transformation(self):
        logging.info("Data Transformation Initiated")
        try:
            train_df = pandas.read_csv(self.train_data_path)
            target_train = train_df["Risk_Flag"]
            test_df = pandas.read_csv(self.test_data_path)
            target_test = test_df["Risk_Flag"]
            logging.info("Read train and test data completed")

            numerical_columns = [
                "Income",
                "Age",
                "CURRENT_HOUSE_YRS",
                "Experience",
            ]
            categorical_columns = ["House_Ownership", "Car_Ownership", "Married/Single"]

            numerical_transformer = Pipeline(
                steps=[
                    ("NumericalImputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ("CategoricalImputer", SimpleImputer(strategy="most_frequent")),
                    ("LabelEncoder", OrdinalEncoder()),
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("Numerical_Transformer", numerical_transformer, numerical_columns),
                    (
                        "Categorical_Transformer",
                        categorical_transformer,
                        categorical_columns,
                    ),
                ]
            )

            train_arr = preprocessor.fit_transform(train_df)
            test_arr = preprocessor.transform(test_df)

            save_obj(preprocessor, self.preprocessed_obj_path)

            logging.info("Data Transformation Succesfully Terminated")
            return (train_arr, target_train, test_arr, target_test, self.preprocessed_obj_path)

        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)