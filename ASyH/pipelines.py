# ASyH Concrete Pipeline definitions

from ASyH.pipeline import Pipeline
from ASyH.models import CopulaGANModel, CTGANModel, GaussianCopulaModel, TVAEModel


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


class CTABGANSynthesizerPipeline(Pipeline):

    def __init__(self, input_data):
        Pipeline.__init__(self,
                        model=CTABGANSynthesizer(data=input_data),
                        input_data=input_data)