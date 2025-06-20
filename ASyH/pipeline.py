# ToDo: detailed class docstring for class Pipeline.
"""ASyH Pipeline base class"""
import tempfile
import os

from ASyH.data import Data
from ASyH.model import Model
from ASyH.abstract_pipeline import AbstractPipeline
from ASyH.hook import ScoringHook, PreprocessHook, PostprocessHook
from ASyH.utils import flatten_dict


class Pipeline(AbstractPipeline):
    """The basic ASyH Pipeline."""

    def __init__(self, model: Model, input_data: Data):
        self._model = model
        self._input_data = input_data
        self._preprocessing_hook = PreprocessHook()
        self._postprocessing_hook = PostprocessHook()
        self._scoring_hook = ScoringHook()

    @property
    def model(self):
        return self._model

    def add_scoring(self, scoring_function):
        self._scoring_hook.add(scoring_function)
    
    def add_preprocessing(self, preprocess_function):
        self._preprocessing_hook.add(preprocess_function)
    
    def add_postprocessing(self, postprocess_function):
        self._postprocessing_hook.add(postprocess_function)

    def run(self):
        save_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as workdir:
            os.chdir(workdir)
            self._input_data = self._preprocessing_hook.execute(self._input_data)
            synthetic_data = Data(data=self._model.synthesize())
            # synthetic_data = self._postprocessing_hook.execute(synthetic_data)
            detailed_scores = self._scoring_hook.execute(self._input_data,
                                                         synthetic_data)
        os.chdir(save_cwd)
        print(f'{self.model.model_type} Scoring: {str(detailed_scores)}')
        # Assuming, the scoring functions are maximizing, nomalized, and
        # weighted equally:
        scores = flatten_dict(detailed_scores)
        return sum(scores.values()) / len(scores)


AbstractPipeline.register(Pipeline)
