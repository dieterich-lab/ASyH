# ASyH Concrete Model-derived classes
#
# ToDos:
#   Implement adapt() for models to tune the model internals to the data.

from ASyH.model import Model
import sdv


class TVAEModel(Model):

    def __init__(self, data=None):
        if data:
            sdv_model = sdv.tabular.TVAE(field_types=data.metadata().metadata)
        else:
            sdv_model = sdv.tabular.TVAE()
        Model.__init__(self, model_type='TVAE',
                       sdv_model=sdv_model, data=data)

    def adapt(self):
        '''Method to adapt the TVAE sdv model internals to data'''
        # for now, just pass


class CTGANModel(Model):

    def __init__(self, data=None):
        if data:
            sdv_model = sdv.tabular.CTGAN(field_types=data.metadata().metadata)
        else:
            sdv_model = sdv.tabular.CTGAN()
        Model.__init__(self, model_type='CTGAN',
                       sdv_model=sdv_model, data=data)

    def adapt(self):
        '''Method to adapt the CTGAN sdv model internals to data'''


class CopulaGANModel(Model):

    def __init__(self, data=None):
        if data:
            sdv_model = sdv.tabular.CopulaGAN(field_types=data.metadata().metadata)
        else:
            sdv_model = sdv.tabular.CopulaGAN()
        Model.__init__(self, model_type='CopulaGAN',
                       sdv_model=sdv_model, data=data)

    def adapt(self):
        '''Method to adapt the CopulaGAN sdv model internals to data'''


class GaussianCopulaModel(Model):

    def __init__(self, data=None):
        if data:
            sdv_model = sdv.tabular.GaussianCopula(field_types=data.metadata().matadata)
        else:
            sdv_model = sdv.tabular.GaussianCopula()
        Model.__init__(self, model_type='GaussianCopula',
                       sdv_model=sdv_model, data=data)

    def adapt(self):
        '''Method to adapt the Gaussian Copula sdv model internals to data'''
