# ToDo: detailed class docstring for class Pipeline.
'''ASyH Pipeline base class'''


from ASyH.abstract_pipeline import AbstractPipeline
from ASyH.hook import ScoringHook
import ASyH.data


class Pipeline(AbstractPipeline):
    '''The basic ASyH Pipeline.'''

    _scoring_hook = ScoringHook()

    @property
    def model(self):
        return self._model

    def add_scoring(self, scoring_function):
        self._scoring_hook.add(scoring_function)

    def __init__(self, model, input_data):
        self._model = model
        self._input_data = input_data

    def _pipeline(self):
        synthetic_data = ASyH.data.Data(data=self._model.synthesize())
        detailed_scores = self._scoring_hook.execute(self._input_data,
                                                     synthetic_data)

        print("Scoring: " + str(detailed_scores))
        # Assuming, the scoring functions are maximizing, nomalized, and
        # weighted equally:
        return sum(detailed_scores.values()) / len(detailed_scores)

    def run(self):
        '''Start the pipeline'''
        return self._pipeline()


AbstractPipeline.register(Pipeline)
