# ASyH Concrete Pipeline definitions

from ASyH.pipeline import Pipeline
from ASyH.models import CopulaGANModel, CTGANModel, GaussianCopulaModel, TVAEModel


class CopulaGANPipeline(Pipeline):

    def _pipeline(self):
        model = CopulaGANModel(data=self._input_data)
        synthesized_data = model.synthesize()
        score = self._scoring_fn(self._input_data, synthesized_data)
        return score


class CTGANPipeline(Pipeline):

    def _pipeline(self):
        model = CTGANModel(data=self._input_data)
        synthesized_data = model.synthesize()
        score = self._scoring_fn(self._input_data, synthesized_data)
        return score


class GaussianCopulaPipeline(Pipeline):

    def _pipeline(self):
        model = CTGANModel(data=self._input_data)
        synthesized_data = model.synthesize()
        score = self._scoring_fn(self._input_data, synthesized_data)
        return score


class TVAEPipeline(Pipeline):

    def _pipeline(self):
        model = TVAEModel(data=self._input_data)
        synthesized_data = model.synthesize()
        score = self._scoring_fn(self._input_data, synthesized_data)
        return score
