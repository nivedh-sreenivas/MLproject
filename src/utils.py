#Used to store common functions used across the project
import os
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(train_x, train_y, test_x, test_y, models):
    try:
        scores = {}

        for i in range(len(list(models))):
            model = list(models.values())[i] #Extract model from dict

            model.fit(train_x, train_y) #Fitting model

            test_y_pred = model.predict(test_x) #test data pred.

            model_score = r2_score(test_y, test_y_pred) #scoring using r2

            scores[list(models.keys())[i]] = model_score

        return scores 
    
    except Exception as e:
        raise CustomException(e, sys)
