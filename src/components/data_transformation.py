import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    
class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def initiate_data_transformation(self,raw):
        logging.info('Starting data transformation')
        
        try:
            # os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            df = pd.read_csv(raw)
            
            
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            
            train_set.to_csv(self.transformation_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.transformation_config.test_data_path,index=False,header=True)
            
            logging.info("train test split complete")
            
            return(
                train_set,
                test_set
            )
            
        except Exception as e:
            raise CustomException(e,sys)
        



