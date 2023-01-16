'''ASyH Concrete Model-Derived Classes'''
#
# ToDos:
#   Implement adapt() for models to tune the model internals to the data.

import sdv
from ASyH.model import Model


class TVAEModel(Model):
    '''Specific ASyH Model for SDV\'s TVAE model.'''

    def __init__(self, data=None):

        def sdv_model_constructor(arg_dict):
            return sdv.tabular.TVAE(**arg_dict)

        Model.__init__(self, model_type='TVAE',
                       sdv_model_constructor=sdv_model_constructor,
                       data=data)

    def adapted_arguments(self, data):
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

    def __init__(self, data=None):

        def sdv_model_constructor(arg_dict):
            return sdv.tabular.CTGAN(**arg_dict)

        Model.__init__(self, model_type='CTGAN',
                       sdv_model_constructor=sdv_model_constructor,
                       data=data)

    def adapted_arguments(self, data):
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

    def __init__(self, data=None):
        def sdv_model_constructor(arg_dict):
            return sdv.tabular.copulagan.CopulaGAN(**arg_dict)

        Model.__init__(self, model_type='CopulaGAN',
                       sdv_model_constructor=sdv_model_constructor,
                       data=data)

    def adapted_arguments(self, data):
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

    def __init__(self, data=None):
        def sdv_model_constructor(arg_dict):
            return sdv.tabular.copulas.GaussianCopula(**arg_dict)

        Model.__init__(self, model_type='GaussianCopula',
                       sdv_model_constructor=sdv_model_constructor,
                       data=data)

    def adapt_arguments(self, data):
        '''Method to adapt the Gaussian Copula sdv model internals to data'''
        return {'field_types': data.metadata.metadata}
