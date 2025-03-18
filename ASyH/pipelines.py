# ASyH Concrete Pipeline definitions

from ASyH.pipeline import Pipeline
from ASyH.models import CopulaGANModel, CTGANModel, GaussianCopulaModel, TVAEModel, CTABGAN_Model

import tempfile
import os
from ASyH.data import Data
from ASyH.hook import ScoringHook, PreprocessHook, PostprocessHook
from ASyH.utils import flatten_dict
import numpy as np
import pandas as pd


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
            # self._input_data = self._preprocessing_hook.execute(self._input_data)
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