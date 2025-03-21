# ASyH Concrete Pipeline definitions
import os
import tempfile
import numpy as np
import pandas as pd
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ASyH.pipeline import Pipeline
from ASyH.models import CopulaGANModel, CTGANModel, GaussianCopulaModel, TVAEModel, CTABGAN_Model

from ASyH.data import Data
from ASyH.hook import ScoringHook, PreprocessHook, PostprocessHook
from ASyH.utils import flatten_dict
# import pdb

# creating a custom logger
logger = logging.getLogger("asyh_logger")
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)

# Create a formatter and attach to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


class CopulaGANPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(self,
                          model=CopulaGANModel(data=input_data),
                          input_data=input_data)


class CTGANPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(self,
                          model=CTGANModel(data=input_data),
                          input_data=input_data)


class GaussianCopulaPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(self,
                          model=GaussianCopulaModel(data=input_data),
                          input_data=input_data)


class TVAEPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(self,
                          model=TVAEModel(data=input_data),
                          input_data=input_data)
        

class CTABGANPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(self,
                          model=CTABGAN_Model(data=input_data),
                          input_data=input_data)
        

    def add_postprocessing(self, postprocess_function):
        self._postprocessing_hook.add(postprocess_function)



    # TODO: later on, wrap the following functions find_closest, generate_col_maps, ... in the static class
    def find_closest(self, value, discrete_vals) -> float:
        idx = np.argmin(np.abs(discrete_vals - value))
        return discrete_vals[idx]
    

    def generate_col_maps(self, data, categoric_cols) -> dict:
        df = data.data

        def make_labels_to_nums(df_col) -> dict:
            unique_labels = df_col.sort_values().unique()
            label2num = {i:label for i,label in enumerate(unique_labels)}
            return label2num
        
        col_maps = {}
        for col in categoric_cols:
            col_maps[col] = make_labels_to_nums(df[col])
        return col_maps
    

    def discretize_column(self, df, col_name, col_maps) -> pd.DataFrame:
        discrete_values = np.array(list(col_maps[col_name].keys()))
        df[col_name] = df[col_name].apply(lambda val: (val/max(df[col_name]) * max(discrete_values)))
        df[col_name] = df[col_name].apply(lambda val: self.find_closest(val, discrete_values))
        df[col_name] = df[col_name].apply(lambda val: col_maps[col_name][val])
        return df
    

    # apply the discretize_column function to all columns of interest in the dataframe df_forest
    def discretize_cols(self, df, categoric_cols, col_maps) -> pd.DataFrame:
        # df = data.data
        for col in categoric_cols:
            df_forest = self.discretize_column(df, col, col_maps)
        return df_forest
    

    def identify_categorical_cols(self, data) -> list:
        # metadata : the metadata object
        # example of metadata content
        # metadata.columns: 
        # {'PATIENT_VISIT_IDENTIFIER': {'sdtype': 'id', 'regex_format': '[0-9]*'}, 'AGE_ABOVE65': {'sdtype': 'categorical'}}
        # find all columns names that are categorical according to the metadata
        metadata = data.metadata
        categoric_cols = [col for col in data.data.columns if metadata.columns[col]['sdtype'] == 'categorical']
        return categoric_cols
    

    def run(self):
        save_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            os.chdir(workdir)
            self._input_data = self._preprocessing_hook.execute(self._input_data)
            # TODO: later add the following few lines to the static class
            synthetic_data_ = Data(data=self._model.synthesize())
            synthetic_data_df = pd.DataFrame(synthetic_data_.data, columns=self._input_data.data.columns)
            categoric_cols =  self.identify_categorical_cols(self._input_data)
            col_maps = self.generate_col_maps(self._input_data, categoric_cols)
            
            synthetic_data_df = self.discretize_cols(synthetic_data_df, categoric_cols, col_maps)
            synthetic_data = Data(synthetic_data_df)
            synthetic_data.set_metadata(self._input_data.metadata)
            # self.add_postprocessing
            # synthetic_data = self._postprocessing_hook.execute(synthetic_data)
            detailed_scores = self._scoring_hook.execute(self._input_data,
                                                         synthetic_data)
        os.chdir(save_cwd)
        print(f'{self.model.model_type} Scoring: {str(detailed_scores)}')
        # Assuming, the scoring functions are maximizing, nomalized, and
        # weighted equally:
        scores = flatten_dict(detailed_scores)
        return sum(scores.values()) / len(scores)


# ASyH static class for preprocessing functions
class Preprocess:

    def __new__(cls, *args, **kwargs):
        raise TypeError("Preprocess class cannot be instantiated.")
    
    @staticmethod
    def normalize(data):
        '''Normalize the data.'''
        return data
    
    # the method that identifies the date columns in the dataframe (entry like '2021-01-01')
    # and returns the list of these columns
    @staticmethod
    def identify_date_cols(data) -> list:
        metadata = data.metadata
        date_cols = [col for col in data.data.columns if metadata.columns[col]['sdtype'] == 'datetime']
        return date_cols
    
    # the method that converts the dates in the dataframe to the number of days since the 1900-01-01
    @staticmethod
    def convert_dates(data, date_cols) -> pd.DataFrame:
        df = data.data
        for col in date_cols:
            df[col] = pd.to_datetime(df[col])
            df[col] = (df[col] - pd.Timestamp("1900-01-01")) // pd.Timedelta('1D')
            # BREAK:
            # import pudb.remote as remote; remote.set_trace()
        return df
    
    # the method that does inverse operation to convert_dates
    @staticmethod
    def convert_back_dates(df, date_cols) -> pd.DataFrame:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col] + pd.Timestamp("1900-01-01"))
        return df
    

    # the function that applies identify_date_cols and convert_dates to the dataframe
    @staticmethod
    def convert_all_dates(data) -> Data:
        metadata = data.metadata
        logger.info(f"Metadata of columns is \n {metadata.columns}")
        date_cols = Preprocess.identify_date_cols(data)
        logger.info(f"Date columns are \n {date_cols}")
        df = Preprocess.convert_dates(data, date_cols)
        logger.info(f"The head of preprocessed data frame is \n {df.head()}")
        data_obj = Data(df, metadata=metadata)
        return data_obj

    
    @staticmethod
    def impute(input_data) -> Data:
        # MICE = IterativeImputer(verbose=False)
        metadata = input_data.metadata
        imputer = IterativeImputer(
            # estimator='RandomForestRegressor',              # Default: BayesianRidge
            estimator=None,
            max_iter=10,                 # Maximum iterations per feature
            tol=0.001,                   # Stopping tolerance threshold
            random_state=42,             # Reproducibility                                                                                     
            initial_strategy='median'      # Initial imputation method
            )
        # set_trace() # breakpoint
        imp_data = imputer.fit_transform(input_data.data)
        imp_data = Data(imp_data, metadata=metadata)
        return imp_data

    
    # its inverse function
    @staticmethod
    def convert_back_all_dates(data) -> Data:
        # FIX:
        # need to use metadata besides dataframe, or a data object
        # cannot identify columns with datetime w/o that
        metadata = data.metadata
        date_cols = Preprocess.identify_date_cols(data)
        df = Preprocess.convert_back_dates(data.data, date_cols)
        data_obj = Data(df, metadata=metadata)
        return data_obj


    # the method that identifies the categorical columns in the dataframe
    @staticmethod
    def identify_categorical_cols(data) -> list:
        metadata = data.metadata
        categoric_cols = [col for col in data.data.columns if metadata.columns[col]['sdtype'] == 'categorical']
        return categoric_cols
    
    
