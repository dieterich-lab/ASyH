'''ASyH Concrete Model-Derived Classes'''
from typing import Optional, Dict, Any

import sdmetrics
import rdt

#
# ToDos:
#   Implement adapt() for models to tune the model internals to the data.

import sdv
from ASyH.data import Data
from ASyH.model import Model


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
                       sdv_model_class=sdv.single_table.CTGANSynthesizer,
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


def _init_iterative_scores(metadata_dict: Dict[str, any]):
    return_dict = {}
    for numerical in metadata_dict.variables_by_type('numerical'):
        return_dict[numerical] = ('', 0.0)
    return_dict['categorical'] = ('', 0.0)
    return return_dict
