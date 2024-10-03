import os, sys

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from src.exception import CustomException
from src.utils import save_obj
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import ModelSelector


@dataclass
class ModelTrainer:
    model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, train_ar, target_train, test_arr, target_test):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_ar,
                target_train,
                test_arr,
                target_test,
            )

            models = {
                "RandomForestClassifier": RandomForestClassifier(),
                "LogisticRegression": LogisticRegression(),
                "XGBClassifier": XGBClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "DecisionTreeClassifier": DecisionTreeClassifier(),
                "KNeighborsClassifier": KNeighborsClassifier(),
            }

            model_selector = ModelSelector(models)

            model_report: dict = model_selector.fit_models(
                X_train, y_train, X_test, y_test
            )
            if model_selector.get_best_score() < 0.6:
                raise CustomException("No model found")
            else:
                logging.info(
                    f"Best model found on both training and testing dataset: {model_selector.get_best_model_name()}"
                )
                save_obj(model_selector.get_best_model(), self.model_path)
                return (
                    model_report,
                    f"{model_selector.get_best_score()} Score for {model_selector.get_best_model_name()} Model and Classification Report is {classification_report(y_test, model_selector.get_best_model().predict(X_test))}",
                )

            # tuned_model,best_param,best_score = model_selector.Tune_best_model(
            #     best_model_name,X_train,y_train
            # )
            # logging.info(f"{best_model_name} Tunned Succecfull")
            # logging.info(f"Best parameters for {best_model_name}: {best_param}")
            # save_obj(tuned_model, self.model_path)
            # return model_report,f"{best_score} Score After Tuning {best_model_name} Model"

        except Exception as e:
            logging.info(f"Exception occured in model training - {e}")
            raise CustomException(e, sys)
