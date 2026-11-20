#Used to store common functions used across the project
import os, sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

# -- Function for SAVING objects --
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


# -- Function for READING objects --
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


# -- MODEL EVALUATION FUNCTION --
def evaluate_models(train_x, train_y, test_x, test_y, models, param):
    try:
        scores = {}

        for i in range(len(list(models))):
            model = list(models.values())[i] # Extract model from dict
            para = param[list(models.keys())[i]] # Extracting respective parameters

            gs = GridSearchCV(model, para, cv = 5, n_jobs= -1) # use n_jobs = -1 to utilize all CPU cores
            gs.fit(train_x,train_y)

            model.set_params(**gs.best_params_) # Adopts best settings
            model.fit(train_x,train_y)

            '''
            print(gs.best_params_)   # best combination found
            print(gs.best_score_)    # its average CV score
            '''

            #model.fit(train_x, train_y) #Fitting model

            test_y_pred = model.predict(test_x) #test data pred.

            model_score = r2_score(test_y, test_y_pred) #scoring using r2

            scores[list(models.keys())[i]] = model_score

        return scores 
    
    except Exception as e:
        raise CustomException(e, sys)