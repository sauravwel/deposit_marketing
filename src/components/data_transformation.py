import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
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

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate','cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
            categorical_columns = [
                'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month', 'day_of_week', 'poutcome',
            ]

            target_columns = ['marital', 'education']
            target_encode = ce.target_encoder.TargetEncoder(cols=target_columns, drop_invariant=True, return_df=False, handle_unknown='value', handle_missing='value', min_samples_leaf=1, smoothing=1.0)


            # define the mapping dictionaries for binary encoding
            contact_mapping = {'cellular': 1, 'telephone': 0}
            loan_mapping = {'yes': 1, 'unknown': 0, 'no': 0}
            housing_mapping = {'yes': 1, 'unknown': 0, 'no': 0}
            default_mapping = {'no': 1, 'unknown': 0, 'yes': 0}
            poutcome_mapping = {'nonexistent': 0, 'failure': 0, 'success': 1}

            target_columns = ['marital', 'education', 'y']

            # function to replace categorical values with binary
            def replace_with_binary(column, mapping,column_name):
                #print(column)
                column = column.replace(mapping)
                #print(f'After {column}')
                return column
                #return column.apply(mapping).astype('uint8')

            def previous_binary(x):
                return (x > 0).astype('uint8')

            def var_rate_transformer(X):
                emp_var_rate = X
                emp_var_rate = np.where(emp_var_rate > 0, emp_var_rate * -0.0001, emp_var_rate * 1)
                emp_var_rate = emp_var_rate * -1
                emp_var_rate = np.where(emp_var_rate < 1, -np.log(emp_var_rate), np.log(emp_var_rate))
                X = emp_var_rate.astype('uint8')
                
                return X

            def keep_column(X):
                return X

            def target_encoding_wrapper(X):
                target_columns = ['marital', 'education']
                #print(X)
                target_encode = ce.target_encoder.TargetEncoder(cols=target_columns, drop_invariant=True, return_df=False, handle_unknown='value', handle_missing='value', min_samples_leaf=1, smoothing=1.0)
                temp = target_encode.fit_transform(X, X['y'])
                #print(temp)
                return temp

            # pipeline for binary encoding of categorical columns
            binary_pipeline = Pipeline([
                ('contact_binary', FunctionTransformer(replace_with_binary, kw_args={'mapping': contact_mapping, 'column_name': 'contact'})),
                ('loan_binary', FunctionTransformer(replace_with_binary, kw_args={'mapping': loan_mapping, 'column_name': 'loan'})),
                ('housing_binary', FunctionTransformer(replace_with_binary, kw_args={'mapping': housing_mapping, 'column_name': 'housing'})),
                ('default_binary', FunctionTransformer(replace_with_binary, kw_args={'mapping': default_mapping, 'column_name': 'default'})),
                ('poutcome_binary', FunctionTransformer(replace_with_binary, kw_args={'mapping': poutcome_mapping, 'column_name': 'poutcome'})),
                ('pdays_binary', FunctionTransformer(lambda x: x.replace(999,0).astype('uint8'))),
                ('previous_binary',  FunctionTransformer(previous_binary))
            ])
            #pipeline for sign change for index
            signchange_pipeline = Pipeline([
                ('confi_idx',FunctionTransformer(lambda x: x.apply(lambda x: -1*x).astype('uint8')))
            ])

            #pipeline for spcial transformations
            numeric_emp_pipeline = Pipeline([
                ('emp_var_rate', FunctionTransformer(var_rate_transformer))
            ])

            # pipeline for scaling numeric columns
            numeric_log_pipeline = Pipeline([
                ('log2', FunctionTransformer(np.log2))
            ])

            
            # target_encoding pipeline
            target_encoder = Pipeline([
                ('target_encoding', FunctionTransformer(target_encoding_wrapper))
            ])

            # pipeline for keeping columns
            numeric_col_pipeline = Pipeline([
                ('keep', FunctionTransformer(keep_column))
            ])


            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # column transformer to apply the pipelines to the respective columns
            preprocessor = ColumnTransformer(transformers=[
                ('binary', binary_pipeline, ['contact', 'loan', 'housing', 'default', 'poutcome','pdays','previous']),
                ('confi_idx', signchange_pipeline, ['cons.conf.idx']),
                ('emp_var', numeric_emp_pipeline, ['emp.var.rate']),
                ('numeric', numeric_log_pipeline, ['nr.employed']),
                ('onehot', OneHotEncoder(handle_unknown='ignore'), ['job', 'month', 'day_of_week']),
                ('column_keep',numeric_col_pipeline,['cons.price.idx','age','euribor3m','campaign']),
                ('target_encoding', target_encoder, target_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name= 'y'
            
            train_df[target_column_name]=train_df[target_column_name].map({'no':0, 'yes':1}).astype('uint8')

            test_df[target_column_name]=test_df[target_column_name].map({'no':0, 'yes':1}).astype('uint8')

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            train_arr=preprocessing_obj.fit_transform(train_df)
            test_arr=preprocessing_obj.transform(test_df)

            train_arr = np.c_[train_arr]
            test_arr = np.c_[test_arr]
            logging.info(f"Saved preprocessing object.")

            # save_object(

            #     file_path=self.data_transformation_config.preprocessor_obj_file_path,
            #     obj=preprocessing_obj

            # )

            return (
                train_arr,
                test_arr,
               # self.data_transformation_config.preprocessor_obj_file_path,
               5
            )
        except Exception as e:
            raise CustomException(e,sys)