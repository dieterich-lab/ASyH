'''ASyH Concrete Model-Derived Classes'''
from typing import Optional, Dict, Any
import re

import sdmetrics
import rdt

#
# ToDos:
#   Implement adapt() for models to tune the model internals to the data.

import sdv
from ASyH.data import Data, Metadata
from ASyH.model import Model, ModelX
from ASyH.ctabgan_synthesizer import CTABGANSynthesizer
from ASyH.transformer_ctabgan import *
from ASyH.utils import Utils
# from forest_data_prep import DataPrep
import time
import subprocess
import datetime

import pandas as pd

RAND_MAX = 65536


def get_metadata_from_data(data):
    return None if data.metadata is None else data.sdv_metadata


class CTGAN2(sdv.single_table.CTGANSynthesizer):
    _model_sdtype_transformers = {
        'categorical': None,
        'boolean': None
    }
    def __init__(self, metadata, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True):
        # def __init__(self, meta, **kwargs):
        #     super(sdv.single_table.CTGANSynthesizer, self).__init__(meta, **kwargs)

        self._model_kwargs = {
            'embedding_dim': embedding_dim,
            'generator_dim': generator_dim,
            'discriminator_dim': discriminator_dim,
            'generator_lr': generator_lr,
            'generator_decay': generator_decay,
            'discriminator_lr': discriminator_lr,
            'discriminator_decay': discriminator_decay,
            'batch_size': batch_size,
            'discriminator_steps': discriminator_steps,
            'log_frequency': log_frequency,
            'verbose': verbose,
            'epochs': epochs,
            'pac': pac,
            'cuda': cuda
        }

        super().__init__(metadata, enforce_min_max_values=True, enforce_rounding=True, locales=None,
                 embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True)


    ## TODO: remake using the module secrets
    def _set_random_state(self):
        # self._model.set_random
        curr_time = datetime.datetime.now()
        # random_state = int(curr_time.timestamp() * 1e+6)
        timestamp_r = str(curr_time.timestamp())[::-1]
        random_a = int(timestamp_r[:6] + timestamp_r[7:])
        checksum = subprocess.check_output(['cksum', '/var/log/lastlog'])
        random_b = int("".join(re.findall(r'\d', str(checksum))))
        random_state = (random_a + random_b) % RAND_MAX
        self._model.set_random_state(random_state)
        self._random_state_set = True


class TVAEModel(Model):
    '''Specific ASyH Model for SDV\'s TVAE model.'''

    def __init__(self, data: Optional[Data] = None, override_args=None):
        Model.__init__(self,
                       sdv_model_class=sdv.single_table.TVAESynthesizer,
                       data=data,
                       override_args=override_args)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the TVAE sdv model internals to the input
        data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_model_class(**adapt_arguments(data)) => adapted SDV model'''

        data_size = len(data.data.columns)
        dim = 2*data_size
        hidden_layer_dims = (dim, dim)
        return {'metadata': get_metadata_from_data(data),
                'compress_dims': hidden_layer_dims,
                'decompress_dims': hidden_layer_dims,
                'embedding_dim': dim}


class CTGANModel(Model):
    '''Specific ASyH Model for SDV\'s CTGAN model.'''

    def __init__(self, data: Optional[Data] = None, override_args=None):
        Model.__init__(self,
                       # sdv_model_class=sdv.single_table.CTGANSynthesizer,
                       sdv_model_class=CTGAN2,
                       data=data,
                       override_args=override_args)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the CTGAN sdv model internals to the
        input data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_model_class(**adapt_arguments(data)) => adapted SDV model'''
        data_size = len(data.data.columns)
        dim = 4*data_size
        hidden_layer_dims = (dim, dim)
        return {'metadata': get_metadata_from_data(data),
                'generator_dim': hidden_layer_dims,
                'discriminator_dim': hidden_layer_dims}


class CopulaGANModel(Model):
    '''Specific ASyH Model for SDV\'s CopulaGAN model.'''

    def __init__(self, data: Optional[Data] = None, override_args=None):
        Model.__init__(self,
                       sdv_model_class=sdv.single_table.copulagan.CopulaGANSynthesizer,
                       data=data,
                       override_args=override_args)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the CopulaGAN sdv model internals to the
        input data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_model_class(**adapt_arguments(data)) => adapted SDV model'''
        data_size = len(data.data.columns)
        dim = 4*data_size
        hidden_layer_dims = (dim, dim)
        return {'metadata': get_metadata_from_data(data),
                'generator_dim': hidden_layer_dims,
                'discriminator_dim': hidden_layer_dims}


class GaussianCopulaModel(Model):
    '''Specific ASyH Model for SDV\'s GaussianCopula model.'''

    class Regressed_GaussianCopulaSynthesizer(sdv.single_table.copulas.GaussianCopulaSynthesizer):
        _model_sdtype_transformers = {'categorical': rdt.transformers.FrequencyEncoder(add_noise=True)}

    def __init__(self, data: Optional[Data] = None, override_args=None):
        Model.__init__(self,
                       sdv_model_class=self.Regressed_GaussianCopulaSynthesizer,
                       data=data,
                       override_args=override_args)
        self._model_type = 'GaussianCopulaSynthesizer'

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Method to adapt the Gaussian Copula sdv model internals to data'''
        args = {'metadata': get_metadata_from_data(data)}
        args.update(self._tune_GCM_distributions(data))
        return args

    def _tune_GCM_distributions(self, data: Data) -> Dict[str, str]:
        best_scores = self._init_iterative_scores(data.metadata)
        numerical_vars = data.metadata.variables_by_type('numerical')
        column_distributions = dict()

        for dist in sdv.single_table.copulas.GaussianCopulaSynthesizer._DISTRIBUTIONS:
            GCM_model = self.Regressed_GaussianCopulaSynthesizer(metadata=get_metadata_from_data(data),
                                                                 default_distribution=dist)
            GCM_model.fit(data.data)
            synth_data = GCM_model.sample(data.data.shape[0])
            sdmetrics_report = sdmetrics.reports.single_table.QualityReport()
            sdmetrics_report.generate(data.data,
                                      synth_data,
                                      data.metadata.metadata,
                                      verbose=False)

            # 'details' is a pandas dataframe:
            details = sdmetrics_report.get_details(property_name='Column Shapes')

            # numerical variables use KSComplement, categorical/booleans use TVComplement:
            # use the average score for categorical variables
            categorical_score = details[details['Metric'] == 'TVComplement']['Score'].mean()
            if categorical_score > best_scores['categorical'][1]:
                best_scores['categorical'] = (dist, categorical_score)
            # we want detailed fitting distributions for numerical variables
            for num_var in numerical_vars:
                score = details[details['Column'] == num_var]['Score'].values[0]
                if score > best_scores[num_var][1]:
                    best_scores[num_var] = (dist, score)
                    # create the override_args
                    column_distributions = {var: best_scores[var][0]
                                            for var in best_scores
                                            if var != 'categorical'}

        return {'numerical_distributions': column_distributions,
                'default_distribution': best_scores['categorical'][0]}
    
    def _init_iterative_scores(self, metadata_dict: Dict[str, any]):
        return_dict = {}
        for numerical in metadata_dict.variables_by_type('numerical'):
            return_dict[numerical] = ('', 0.0)
        return_dict['categorical'] = ('', 0.0)
        return return_dict


class ForestFlowModel(Model):
    '''Specific ASyH Model for Samsung ForestFlow model.'''
    def __init__(self, data=Optional[Data], override_args=None):
        Model.__init__(self,
                      sdv_model_class=CTABGANSynthesizer,
                      data=data,
                      override_args=override_args)
        self.data = data
        self._model_type = 'ForestFlowSynthesizer'
        self._trained = False
        # the columns which will be temporarily removed from the data until inverse transformation
        self.hidden_columns = [col for col in data.data.columns if '_u' in col]
        self.data_hidden = None
        

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the CTGAN sdv model internals to the
        input data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_model_class(**adapt_arguments(data)) => adapted SDV model'''
        data_size = len(data.data.columns)
        dim = 4 * data_size
        hidden_layer_dims = (dim, dim)
        return {'metadata': get_metadata_from_data(data),
                'generator_dim': hidden_layer_dims,
                'discriminator_dim': hidden_layer_dims}
    

    def transform_data_prep(self, data: Data) -> Data:
        # remove hidden columns from the data and store them in a separate dataframe
        self.data_hidden = data.data[self.hidden_columns]
        df_data = data.data.drop(columns=self.hidden_columns, axis=1)
        data = Data(df_data, metadata=data.metadata)

        data = Utils.convert_all_dates(data)

        # check if the dataframe df contains nans
        df = data.data

        if df.isnull().values.any():
            # raise the warning
            Warning('DataFrame contains NaN values. Filling with 0.')
            # impute the missing values using Utils.impute
            data = Utils.impute(data)
        # check if the dataframe df contains infinite values

        # if np.isinf(df.values).any():
        #     # raise the warning
        #     Warning('DataFrame contains infinite values. Filling with 0.')
        #     # replace the infinite values with nan
        #     df.replace([np.inf, -np.inf], np.nan, inplace=True)
        #     # impute the missing values using Utils.impute
        #     data = Utils.impute(data)

        return data
    

    def transform_data_inverse(self, data: Data, src_data: Data) -> Data:
        meta = src_data.metadata
        cat_cols = []

        # # check if data is an instance of DataFrame
        # if not isinstance(data, pd.DataFrame) and isinstance(data, np.ndarray):
        #     # make it a datafame from numpy
        #     data = pd.DataFrame(data, columns=src_data.data.columns)

        for col,property in meta.columns.items():                                                                                     
            if property['sdtype'] != 'numerical':                                                                                        
                cat_cols.append(col)

        col_maps = Utils.generate_col_maps(src_data, cat_cols)
        # data_fake = Data(data, metadata=Metadata(meta))
        data_fake = data

        # df_fake2 = df_fake.copy()

        # data_fake = Utils.convert_back_all_dates(data_fake)

        df_fake2 = Utils.convert_types_pandas(data_fake, meta)

        # df_fake2 = data_fake2.data
        for col in col_maps.keys():
            df_fake2 = Utils.discretize_column(df_fake2, col, col_maps)

        # return hidden columns to the dataframe df_fake2 using self.data_hidden
        df_fake2 = pd.concat([df_fake2, self.data_hidden], axis=1)
        
        synthetic_data = Data(df_fake2, metadata=src_data.metadata)
        data_inverse = Utils.convert_back_all_dates(synthetic_data)
        return data_inverse
    

    def synthesize(self, sample_size: int = -1, 
                   data=None) -> pd.DataFrame:
        '''Create synthetic data.'''
        if data is None:
            data = self.data
        # check if data is an instance of Data
        assert isinstance(data, Data), 'Data is not an instance of Data'

        # transform data
        data_ = self.transform_data_prep(data)

        if not self._trained:
            import ipdb; ipdb.set_trace()
            self._train(data=data_)
        if sample_size == -1:
            sample_size = self._input_data_size

        df_synth = self.sdv_model.sample(sample_size)

        # check if data is an instance of DataFrame
        if not isinstance(df_synth, pd.DataFrame) and isinstance(df_synth, np.ndarray):
            # make it a datafame from numpy
            df_synth = pd.DataFrame(df_synth, columns=data.data.columns)

        data_synth_raw = Data(df_synth, metadata=data.metadata)
        # inverse transform data
        data_out = self.transform_data_inverse(data_synth_raw, data)
        return data_out.data
    

    # def _train(self, data: Optional[Data] = None):
    #     # transform data
    #     data = self.transform_data_prep(data)

    #     if data is None:
    #         data = self._training_data

    #     if hasattr(self._sdv_model, 'add_constraints'):
    #         if self._constraints is not None:
    #             self.sdv_model.add_constraints(self._constraints)

    #     self.sdv_model.fit(data.data)
    #     self._input_data_size = data.data.shape[0]
    #     self._trained = True


# class CTABGAN():
#     def __init__(self,
#                  pd_data,
#                  categorical_columns = [], 
#                  log_columns = [],
#                  mixed_columns= {},
#                  general_columns = [],
#                  non_categorical_columns = [],
#                  integer_columns = []):
#         self.synthesizer = CTABGANSynthesizer()
#         self.raw_df = pd_data
#         self.categorical_columns = categorical_columns
#         self.log_columns = log_columns
#         self.mixed_columns = mixed_columns
#         self.general_columns = general_columns
#         self.non_categorical_columns = non_categorical_columns
#         self.integer_columns = integer_columns

#     def fit(self):
#         start_time = time.time()
#         self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns)
#         self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
#         general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"])
#         end_time = time.time()
#         print('Finished training in',end_time-start_time," seconds.")

#     def generate_samples(self):

#         sample = self.synthesizer.sample(len(self.raw_df))
#         sample_df = self.data_prep.inverse_prep(sample)

#         return sample_df


# def _init_iterative_scores(metadata_dict: Dict[str, any]):
#     return_dict = {}
#     for numerical in metadata_dict.variables_by_type('numerical'):
#         return_dict[numerical] = ('', 0.0)
#     return_dict['categorical'] = ('', 0.0)
#     return return_dict

class CPARModel(Model):
    '''
    Specific ASyH Model for SDV\'s CPAR model,
    responsible for longitudinal multi-variate data generation.
    '''

    def __init__(self, data: Optional[Data] = None, override_args=None):
        Model.__init__(self,
                       sdv_model_class=sdv.sequential.PARSynthesizer,
                       data=data,
                       override_args=override_args)
        self.data = data
        self._model_type = 'CPARSynthesizer'
        self._trained = False
        # the columns which will be temporarily removed from the data until inverse transformation
        # self.hidden_columns = [col for col in data.data.columns if '_u' in col]
        # self.data_hidden = None


    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the CopulaGAN sdv model internals to the
        input data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_model_class(**adapt_arguments(data)) => adapted SDV model'''
        # data_size = len(data.data.columns)
        metadata = get_metadata_from_data(data)
        if isinstance(metadata, Metadata):
            metadata = metadata.metadata
        else:
            assert isinstance(metadata, sdv.metadata.SingleTableMetadata), \
                'metadata should be an instance of SingleTableMetadata'
        # dim = 4 * data_size
        # hidden_layer_dims = (dim, dim)
        return {
                'metadata': metadata,
                'verbose': True,
                'epochs': 100,
                # 'context_columns': ['']
                'enforce_min_max_values': True,
                'cuda': False
                }