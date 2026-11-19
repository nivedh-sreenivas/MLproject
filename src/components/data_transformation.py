import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    # .pkl -> pickle file. used for saving any ML models
    # It's a form of model file . Others include .safetensors, .onnx or .h5

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        # we will use this to store preprocessor variable above
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            #doing separate loggers here to avoid two lines in a log. Looks cleaner
            logging.info(f"Categorical columns: {categorical_features}")
            logging.info(f"Numerical columns: {numerical_features}")
            
            num_pipeline = Pipeline(
                steps = [ #transformers keyword is optional here since its default
                    ("impute", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)) 
                ]
            )
            logging.info("Numerical and Categorical features encoded")

            preprocessor = ColumnTransformer(
                transformers= [ #transformers keyword is optional here since its default
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data have been read.")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Preprocessor object obtained.")

            y = 'math_score' #target feature
            numerical_features = ['reading_score', 'writing_score']

            train_x = train_df.drop(y, axis = 1)
            train_y = train_df[y]

            test_x = test_df.drop(y, axis = 1)
            test_y = test_df[y]

            train_x_data = preprocessing_obj.fit_transform(train_x)
            test_x_data = preprocessing_obj.transform(test_x) #No fitting so that test data dont leak into preprocessing

            # Creating final dataframes where the input features are transformed and combined with respective target feature
            train_data = np.c_[train_x_data, train_y]
            test_data  = np.c_[test_x_data, test_y]
            # c_ is used to join numpy arrays. If pandas dataframe, then we can use pd.concat 

            logging.info("Train and test data preprocessed and saved.")

            #Saving fitted preprocessor into a file object
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_data,
                test_data,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            #preprocessor object passed since it contains the .pkl file needed 

        except Exception as e:
            raise CustomException(e, sys)


