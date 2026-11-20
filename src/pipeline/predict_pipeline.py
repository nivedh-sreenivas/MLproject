import os, sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from dataclasses import dataclass, asdict

class PredictPipeline:
    def __init__(self): #Done so that below files have to be loaded once rather than for each prediction
        # Loading artifacts
        model_path = os.path.join("artifacts","model.pkl")
        preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

        #Loading mlm objects
        self.model = load_object(file_path=model_path)
        self.preprocessor = load_object(file_path=preprocessor_path)

    def predict(self, features):
        try:
            #Data transformation
            data_scaled = self.preprocessor.transform(features)
            
            #Prediction
            preds = self.model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)



@dataclass
class CustomData: # Will be responsible for mapping all the inputs given from html to back-end 
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int

    def get_data_as_data_frame(self):
        return pd.DataFrame([asdict(self)])

        '''
        will create dictionary given below:

        { "gender": [self.gender],
        "race_ethnicity": [self.race_ethnicity],
        "parental_level_of_education": [self.parental_level_of_education],
        "lunch": [self.lunch],
        "test_preparation_course": [self.test_preparation_course],
        "reading_score": [self.reading_score],
        "writing_score": [self.writing_score] }
        
        '''