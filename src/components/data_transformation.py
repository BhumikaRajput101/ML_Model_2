import sys 
from dataclasses  import dataclass
import os

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging

from src.pipeline.utils import save_obj


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprossor.pkl')

class DataTransformation:
    def __init__(self):
        self.DataTransformationconfig=DataTransformationconfig()

    def get_data_tranformer_object(self):
        try:
            numerical_column=['writing score','reading score']
            cateorical_column=['gender','race/ethnicity','parental level of education','lunch',
                               'test preparation course']
            
            num_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])

            cat_pipeline=Pipeline(

                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_hot_encoder",OneHotEncoder(handle_unknown='ignore')),
                   

                ]


            )
            logging.info("numercial and categorical colunms encoding comleted")

            preprocessor=ColumnTransformer(
                [('num_pipeline',num_pipeline,numerical_column),
                 ("cat_pipeline",cat_pipeline,cateorical_column)
                 
                 ]
            )
            return preprocessor

            
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("read train test data completed")
            logging.info("Obtaining preproessing object")

            preprocessing_obj=self.get_data_tranformer_object()
            target_column='math score'
            numerical_column=['writing score','reading score']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]
            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying preprocessing on test an train data")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object.")
            save_obj(
                file_path=self.DataTransformationconfig.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return(
                train_arr,test_arr,self.DataTransformationconfig.preprocessor_obj_file_path,
            )
        




            
        except Exception as e:
            raise CustomException(e,sys)
        

 


        
    
        

    
                                   