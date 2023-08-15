import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score      

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evalute_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('splitting training and testing data')
            x_train,y_train,x_test,y_test=(
                train_arr.drop('target',axis=1),
                train_arr['target'],
                test_arr.drop('target',axis=1),
                test_arr['target']
            )
            
            # Put Models into a dictionary
            models = {"Logistic Regression" : LogisticRegression(),
                    "KNN": KNeighborsClassifier(),
                    "Random Forest": RandomForestClassifier(),
                    "svm": LinearSVC(),
                    "Naive Bayes": GaussianNB()}
            
            model_report:dict = evalute_models(x_train,y_train,x_test,y_test,models)
            
            ## to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))
            
            ## Get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            logging.info("Best model found")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(x_test)
            accuracy = accuracy_score(y_test,predicted)
            
            return accuracy
        
        
        except Exception as e:
            raise CustomException(e,sys)