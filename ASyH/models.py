'''ASyH Concrete Model-Derived Classes'''
from typing import Optional, Dict, Any

#
# ToDos:
#   Implement adapt() for models to tune the model internals to the data.

import sdv
from ASyH.data import Data
from ASyH.model import Model


class TVAEModel(Model):
    '''Specific ASyH Model for SDV\'s TVAE model.'''

    def sdv_model_constructor(self, arg_dict):
        return sdv.tabular.TVAE(**arg_dict)

    def __init__(self, data: Optional[Data] = None):
        Model.__init__(self, model_type='TVAE',
                       data=data)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the TVAE sdv model internals to the input
        data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_constructor(**adapt_arguments(data)) => adapted SDV model'''

        data_size = len(data.data.columns)
        dim = 2*data_size
        hidden_layer_dims = (dim, dim)
        return {'field_types': _get_field_types_from_data(data),
                'compress_dims': hidden_layer_dims,
                'decompress_dims': hidden_layer_dims,
                'embedding_dim': dim}


def _get_field_types_from_data(data):
    return None if data.metadata is None else data.metadata.metadata


class CTGANModel(Model):
    '''Specific ASyH Model for SDV\'s CTGAN model.'''

    def sdv_model_constructor(self, arg_dict):
        return sdv.tabular.CTGAN(**arg_dict)

    def __init__(self, data: Optional[Data] = None):
        Model.__init__(self, model_type='CTGAN',
                       data=data)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the CTGAN sdv model internals to the
        input data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_constructor(**adapt_arguments(data)) => adapted SDV model'''
        data_size = len(data.data.columns)
        dim = 4*data_size
        hidden_layer_dims = (dim, dim)
        return {'field_types': _get_field_types_from_data(data),
                'generator_dim': hidden_layer_dims,
                'discriminator_dim': hidden_layer_dims}


class CopulaGANModel(Model):
    '''Specific ASyH Model for SDV\'s CopulaGAN model.'''

    def sdv_model_constructor(self, arg_dict):
        return sdv.tabular.copulagan.CopulaGAN(**arg_dict)

    def __init__(self, data: Optional[Data] = None):
        Model.__init__(self, model_type='CopulaGAN',
                       data=data)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Create SDV model specific argument dict to pass to the constructor.
        This method is meant to adapt the CopulaGAN sdv model internals to the
        input data.
        The method returns a dict of keyword arguments to be passed to the
        specific SDV model constructor with the ** operator:
        sdv_constructor(**adapt_arguments(data)) => adapted SDV model'''
        data_size = len(data.data.columns)
        dim = 4*data_size
        hidden_layer_dims = (dim, dim)
        return {'field_types': _get_field_types_from_data(data),
                'generator_dim': hidden_layer_dims,
                'discriminator_dim': hidden_layer_dims}


class GaussianCopulaModel(Model):
    '''Specific ASyH Model for SDV\'s GaussianCopula model.'''

    def sdv_model_constructor(self, arg_dict):
        return sdv.tabular.copulas.GaussianCopula(**arg_dict)

    def __init__(self, data: Optional[Data] = None):
        Model.__init__(self, model_type='GaussianCopula',
                       data=data)

    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Method to adapt the Gaussian Copula sdv model internals to data'''
        return {'field_types': _get_field_types_from_data(data)}
