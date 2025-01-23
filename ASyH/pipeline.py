# ToDo: detailed class docstring for class Pipeline.
"""ASyH Pipeline base class"""
import tempfile
import os

from ASyH.data import Data
from ASyH.model import Model
from ASyH.abstract_pipeline import AbstractPipeline
from ASyH.hook import ScoringHook, PreprocessHook
from ASyH.utils import flatten_dict


class Pipeline(AbstractPipeline):
    """The basic ASyH Pipeline."""

    def __init__(self, model: Model, input_data: Data):
        self._model = model
        self._input_data = input_data
        self._preprocessing_hook = PreprocessHook()
        self._scoring_hook = ScoringHook()

    @property
    def model(self):
        return self._model

    def add_scoring(self, scoring_function):
        self._scoring_hook.add(scoring_function)

    def run(self):
        save_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            os.chdir(workdir)
            self._input_data = self._preprocessing_hook.execute(self._input_data)
            synthetic_data = Data(data=self._model.synthesize())
            detailed_scores = self._scoring_hook.execute(self._input_data,
                                                         synthetic_data)
        os.chdir(save_cwd)
        print(f'{self.model.model_type} Scoring: {str(detailed_scores)}')
        # Assuming, the scoring functions are maximizing, nomalized, and
        # weighted equally:
        scores = flatten_dict(detailed_scores)
        return sum(scores.values()) / len(scores)


AbstractPipeline.register(Pipeline)
