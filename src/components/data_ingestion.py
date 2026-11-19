import os
import sys #for CustomException

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig

import pandas as pd #for df
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # powerful tool. will be explained later. used for class variables

@dataclass #decorator used. Removes need for init in class. We can directly define class var.
class DataIngestionConfig:
    '''
    all the inputs needed for data config will be done within this class
    (similar practice will be followed for data_transformation)
    '''

    train_data_path: str = os.path.join('artifacts',"train.csv")
    
    '''
    train_data_path is of 'str' type
    This is where data ingestion output will be stored. Artifacts folder, file: train.csv
    '''

    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"raw.csv")

#No decorator here since we have functions in below class.
#Use decorator only if class consists of variables.
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        used rn simply for reading dataset
        '''
        logging.info("Enter data ingestion method/component")
        try:
            '''IMPORT DATA'''
            df = pd.read_csv('notebook\data\stud.csv') #Here we input where to read data from
            logging.info('Read the dataset as dataframe') #Keep logging at every stage so it makes debugging easier
            
            '''MAKE DESTINATION FOLDER'''
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            #exist_ok wont make dir again if folder exists

            '''RAW DATA FILE CREATED'''
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            
            '''TRAIN_TEST_SPLIT'''
            logging.info("Train test split initiated")
            train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)

            '''SAVING THE SPLITS INTO SEPARATE FILES'''
            train_data.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Data Ingestion completed")

            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)