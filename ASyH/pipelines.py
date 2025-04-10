# ASyH Concrete Pipeline definitions
import os
import tempfile
import numpy as np
import pandas as pd
import logging
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from ASyH.pipeline import Pipeline
from ASyH.models import CopulaGANModel, CTGANModel, GaussianCopulaModel, TVAEModel, ForestFlowModel

from ASyH.data import Data, Metadata
from ASyH.hook import ScoringHook, PreprocessHook, PostprocessHook
from ASyH.utils import flatten_dict, Utils
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

    def __init__(self, input_data, override_args={"constraints": None}):
        super().__init__(model=CopulaGANModel(data=input_data,
                                               override_args=override_args),
                          input_data=input_data)


class CTGANPipeline(Pipeline):

    def __init__(self, input_data, override_args={"constraints": None}):
        super().__init__(self,
                          model=CTGANModel(data=input_data,
                                           override_args=override_args),
                          input_data=input_data)


class GaussianCopulaPipeline(Pipeline):

    def __init__(self, input_data, override_args={"constraints": None}):
        super().__init__(model=GaussianCopulaModel(data=input_data,
                                                    override_args=override_args),
                          input_data=input_data)


class TVAEPipeline(Pipeline):

    def __init__(self, input_data, override_args={"constraints": None}):
        super().__init__(model=TVAEModel(data=input_data,
                                          override_args=override_args),
                          input_data=input_data)
        

class ForestFlowPipeline(Pipeline):

    def __init__(self, input_data, override_args={"constraints": None}):
        super().__init__(model=ForestFlowModel(data=input_data,
                                              override_args=override_args),
                          input_data=input_data)

    def add_postprocessing(self, postprocess_function):
        self._postprocessing_hook.add(postprocess_function)
    

    # def run(self):
    #     save_cwd = os.getcwd()
    #     with tempfile.TemporaryDirectory() as workdir:
    #         os.chdir(workdir)
    #         self._input_data = self._preprocessing_hook.execute(self._input_data)
    #         # TODO: later add the following few lines to the static class
    #         synthetic_data_ = Data(data=self._model.synthesize())
    #         synthetic_data_df = pd.DataFrame(synthetic_data_.data, columns=self._input_data.data.columns)
    #         categoric_cols =  self.identify_categorical_cols(self._input_data)
    #         col_maps = self.generate_col_maps(self._input_data, categoric_cols)
            
    #         synthetic_data_df = self.discretize_cols(synthetic_data_df, categoric_cols, col_maps)
    #         synthetic_data = Data(synthetic_data_df)
    #         synthetic_data.set_metadata(self._input_data.metadata)
    #         # self.add_postprocessing
    #         # synthetic_data = self._postprocessing_hook.execute(synthetic_data)
    #         detailed_scores = self._scoring_hook.execute(self._input_data,
    #                                                      synthetic_data)
    #     os.chdir(save_cwd)
    #     print(f'{self.model.model_type} Scoring: {str(detailed_scores)}')
    #     # Assuming, the scoring functions are maximizing, nomalized, and
    #     # weighted equally:
    #     scores = flatten_dict(detailed_scores)
    #     return sum(scores.values()) / len(scores)


    def run(self):
        save_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            df_int_cols = Utils.identify_float_columns_that_are_integers(self._input_data.data)
            df_bool_cols = Utils.identify_float_columns_that_are_booleans(self._input_data.data)

            meta = self._input_data.metadata
            src_medset_raw = Data(self._input_data.data, metadata=Metadata(meta))

            src_medset = Utils.convert_all_dates(src_medset_raw)
            df_src = src_medset.data
            # ipdb.set_trace()


            for col in df_src.columns:
                if meta['columns'][col]['sdtype'] == 'numerical':
                    if col in df_int_cols:
                        # df_medset[col] = df_medset[col].astype(int)
                        meta['columns'][col]['computer_representation'] = 'Int'
                    elif col in df_bool_cols:
                        meta['columns'][col]['sdtype'] = 'boolean'
                    else:
                        meta['columns'][col]['computer_representation'] = 'Float'
                # else:
                #     meta['columns'][col]['sdtype'] = 'boolean'

            # with open('meta_updated_medset.json', 'w') as fl_meta:
            #     json.dump(meta, fl_meta)

            # X = df_medset.values
            X = df_src.to_numpy()
            print(X)
            # ipdb.set_trace()

            self.model._train(X)

            Xy_fake = self.model.synthesize(sample_size=X.shape[0])

            print(f"Regression problem: \n {Xy_fake}")

            # with open('fake_medset.npy', 'wb') as np_fl:
            #     np.save(np_fl, Xy_fake)

            df_fake = pd.DataFrame(Xy_fake, columns=df_src.columns)


            # validate the raw dataframe df_fake
            # check if there are empty columns
            empty_cols = df_fake.columns[df_fake.isna().all()]
            if len(empty_cols) > 0:
                print(f"Warning: Found empty columns: {list(empty_cols)}")

            # ipdb.set_trace()

            # Save the raw dataframe before conversion
            # df_fake.to_csv('fake_medset_raw.csv', index=False)

            # df_fake = convert_types_pandas(df_fake, meta)

            data_medset = Data(df_src, metadata=Metadata(meta))

            cat_cols = []
            for col,property in meta['columns'].items():                                                                                     
                if property['sdtype'] == 'categorical':                                                                                        
                    cat_cols.append(col)

            # Ensure cat_cols is not empty
            if not cat_cols:
                print("Warning: No categorical columns found in the metadata")

            col_maps = Utils.generate_col_maps(data_medset, cat_cols)


            data_fake = Data(df_fake, metadata=Metadata(meta))

            # df_fake2 = df_fake.copy()

            data_fake = Utils.convert_back_all_dates(data_fake)

            df_fake2 = self.convert_types_pandas(data_fake.data, meta)

            # df_fake2 = data_fake2.data
            for col in col_maps.keys():
                df_fake2 = self.discretize_column(df_fake2, col, col_maps)
            
            synthetic_data = Data(df_fake2, metadata=self._input_data.metadata)
            # synthetic_data.set_metadata(self._input_data.metadata)
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