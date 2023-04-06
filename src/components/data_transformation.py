import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
import category_encoders as ce

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(data):
        '''
        This function is responsible for data trnasformation
        
        '''
        try:
            # Convert target variable into numeric
            data.y = data.y.map({'no':0, 'yes':1}).astype('uint8')

            # Replacing values with binary ()
            data.contact = data.contact.map({'cellular': 1, 'telephone': 0}).astype('uint8') 
            data.loan = data.loan.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')
            data.housing = data.housing.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')
            data.default = data.default.map({'no': 1, 'unknown': 0, 'yes': 0}).astype('uint8')
            data.pdays = data.pdays.replace(999, 0) # replace with 0 if not contact 
            data.previous = data.previous.apply(lambda x: 1 if x > 0 else 0).astype('uint8') # binary has contact or not

            # binary if were was an outcome of marketing campane
            data.poutcome = data.poutcome.map({'nonexistent':0, 'failure':0, 'success':1}).astype('uint8') 

            # change the range of Var Rate
            data['emp.var.rate'] = data['emp.var.rate'].apply(lambda x: x*-0.0001 if x > 0 else x*1)
            data['emp.var.rate'] = data['emp.var.rate'] * -1
            data['emp.var.rate'] = data['emp.var.rate'].apply(lambda x: -np.log(x) if x < 1 else np.log(x)).astype('uint8')

            # Multiply consumer index 
            data['cons.price.idx'] = (data['cons.price.idx'] * 10).astype('uint8')

            # change the sign (we want all be positive values)
            data['cons.conf.idx'] = data['cons.conf.idx'] * -1

            # re-scale variables
            data['nr.employed'] = np.log2(data['nr.employed']).astype('uint8')
            data['cons.price.idx'] = np.log2(data['cons.price.idx']).astype('uint8')
            data['cons.conf.idx'] = np.log2(data['cons.conf.idx']).astype('uint8')
            data.age = np.log(data.age)

            # less space
            data.euribor3m = data.euribor3m.astype('uint8')
            data.campaign = data.campaign.astype('uint8')
            data.pdays = data.pdays.astype('uint8')

            # fucntion to One Hot Encoding
            def encode(data, col):
                return pd.concat([data, pd.get_dummies(col, prefix=col.name)], axis=1)

            # One Hot encoding of 3 variable 
            data = encode(data, data.job)
            data = encode(data, data.month)
            data = encode(data, data.day_of_week)

            # Drop tranfromed features
            data.drop(['job', 'month', 'day_of_week'], axis=1, inplace=True)

            # Drop the dublicates
            data.drop_duplicates(inplace=True)

            def duration(data):
                data.loc[data['duration'] <= 102, 'duration'] = 1
                data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration'] = 2
                data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration'] = 3
                data.loc[(data['duration'] > 319) & (data['duration'] <= 645), 'duration'] = 4
                data.loc[data['duration']  > 645, 'duration'] = 5
                return data
            duration(data);

            # Target encoding for two categorical feature '''
            # save target variable before transformation
            y = data.y
            # Create target encoder object and transoform two value
            target_encode = ce.target_encoder.TargetEncoder(cols=['marital', 'education']).fit(data, y)
            numeric_dataset = target_encode.transform(data)
            # drop target variable
            numeric_dataset.drop('y', axis=1, inplace=True)

            return data

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            logging.info("Read train and test data completed")
            train_df=pd.read_csv(train_path, sep=';')
            test_df=pd.read_csv(test_path, sep=';')

            logging.info(
                f"Applying preprocessing on training dataframe and testing dataframe."
            )

            train_arr= get_data_transformer_object(train_df)
            test_arr= get_data_transformer_object(test_df)


            logging.info(f"Preprocessing Done.")

            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e,sys)