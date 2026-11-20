import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_data, test_data):
        #we will get train data and test data from transformation.
        try:
            logging.info("Re-splitting train and test data for training now.")
            
            train_x, train_y, test_x, test_y = (
                train_data[:, :-1],
                train_data[:, -1],
                test_data[:, :-1],
                test_data[:, -1]
            )

            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            models_score:dict = evaluate_models(train_x = train_x, train_y = train_y, test_x = test_x, test_y = test_y, models = models)

            # --BEST MODEL--
            best_model_name = max(models_score, key=models_score.get)
            best_model_score = models_score[best_model_name]

            if best_model_score < 0.6: #0.6 -> 60% score
                raise CustomException("Best model score not good enough.")

            logging.info("Best model found.")

            #Saving trainer object that is created
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model_name
            )

            return best_model_score
        
        except Exception as e:
            raise CustomException(e,sys)