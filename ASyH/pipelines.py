# ASyH Concrete Pipeline definitions

from ASyH.pipeline import Pipeline
from ASyH.models import CopulaGANModel, CTGANModel, GaussianCopulaModel, TVAEModel


class CopulaGANPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(CopulaGANModel(data=input_data), input_data)


class CTGANPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(CTGANModel(data=input_data), input_data)


class GaussianCopulaPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(GaussianCopulaModel(data=input_data), input_data)


class TVAEPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(TVAEModel(data=input_data), input_data)
