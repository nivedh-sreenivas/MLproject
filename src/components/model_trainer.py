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
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            #--Hyperparameter tuning--
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [3, 5, 10, None],
                },
                "Random Forest":{
                    'n_estimators': [64,128,256],
                    'max_depth': [None, 5, 10]
                },
                "Gradient Boosting":{
                    'learning_rate': [0.1, 0.05, 0.01],
                    'subsample': [0.7, 0.85, 1.0],
                    'n_estimators': [64, 128, 256],
                    'max_depth': [2, 3, 5],
                },
                "Linear Regression":{},
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2],
                    'algorithm': ['auto'],
                },
                "XGBRegressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'n_estimators': [100, 200, 300, 500], # tree sizes 8, 16, 32 too low and dont perform well on average
                    'max_depth': [2, 3, 4, 6],
                    'subsample': [0.7, 0.85, 1.0],
                    'colsample_bytree': [0.7, 1.0],
                },
                "CatBoosting Regressor": {
                    'depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'iterations': [100, 300, 500],
                    'l2_leaf_reg': [1, 3, 5, 9],
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                    'n_estimators': [50, 100, 200, 300],
                    'loss': ['linear', 'square', 'exponential'],
                }
            }

            models_score:dict = evaluate_models(train_x = train_x, train_y = train_y, 
                                                test_x = test_x, test_y = test_y, 
                                                models = models, param = params)

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

params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }