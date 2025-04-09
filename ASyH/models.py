'''ASyH Concrete Model-Derived Classes'''
from typing import Optional, Dict, Any
import re

import sdmetrics
import rdt

#
# ToDos:
#   Implement adapt() for models to tune the model internals to the data.

import sdv
from ASyH.data import Data
from ASyH.model import Model, ModelX
from ASyH.ctabgan_synthesizer import CTABGANSynthesizer
from ASyH.transformer_ctabgan import *
# from forest_data_prep import DataPrep
import time
import subprocess
import datetime

RAND_MAX = 65536


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
        return {'metadata': _get_metadata_from_data(data),
                'compress_dims': hidden_layer_dims,
                'decompress_dims': hidden_layer_dims,
                'embedding_dim': dim}


def _get_metadata_from_data(data):
    return None if data.metadata is None else data.sdv_metadata


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
        return {'metadata': _get_metadata_from_data(data),
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
        return {'metadata': _get_metadata_from_data(data),
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
        args = {'metadata': _get_metadata_from_data(data)}
        args.update(self._tune_GCM_distributions(data))
        return args

    def _tune_GCM_distributions(self, data: Data) -> Dict[str, str]:
        best_scores = _init_iterative_scores(data.metadata)
        numerical_vars = data.metadata.variables_by_type('numerical')
        column_distributions = dict()

        for dist in sdv.single_table.copulas.GaussianCopulaSynthesizer._DISTRIBUTIONS:
            GCM_model = self.Regressed_GaussianCopulaSynthesizer(metadata=_get_metadata_from_data(data),
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


class ForestFlowModel(Model):
    def __init__(self, data=Optional[Data], override_args=None):
        Model.__init__(self,
                      sdv_model_class=CTABGANSynthesizer,
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
        return {'metadata': _get_metadata_from_data(data),
                'generator_dim': hidden_layer_dims,
                'discriminator_dim': hidden_layer_dims}


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
