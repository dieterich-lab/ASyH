'''Define the standard application of ASyH.'''
import pathlib

from sdv.metrics.tabular import KSComplement, CSTest, CorrelationSimilarity

from ASyH.data import Data
from ASyH.metadata import Metadata
from ASyH.pipelines \
    import CopulaGANPipeline, TVAEPipeline, \
    CTGANPipeline, GaussianCopulaPipeline
from ASyH.dispatch import concurrent_dispatch

import ASyH.metrics.anonymity
import ASyH.metrics


class Application:
    '''The standard application of ASyH.'''

    @property
    def model(self):
        return self._best

    def __init__(self):
        self._results = []
        self._best = None

    def _add_scoring(self, scoring_function, pipelines=None):
        '''Add a scoring function to all pipelines.'''
        if pipelines is None:
            pass
        for pipeline in pipelines:
            pipeline.add_scoring(scoring_function)

    def _select_best(self, results, pipelines=None):
        '''Select the best-scoring model'''
        if pipelines is None:
            return None
        best_score = results.index(max(results))
        self._best = pipelines[best_score].model
        return self._best

    def synthesize(self, input_file,
                   metadata=None, metadata_file=None, sample_size=-1):
        '''Synthesize data using the best-scoring model.'''
        if self._best is None:
            self.process(input_file,
                         metadata_file=metadata_file,
                         metadata=metadata)
        return self._best.synthesize(sample_size)

    def process(self, input_file, metadata_file=None, metadata=None):
        '''Process the default ASyH pipeline.'''

        if metadata_file is not None and metadata is not None:
            Warning('ASyH.App.Application: both metadata_file and metadata have been \
            specified:\n\tDefaulting to use metadata_file as input source.')

        if metadata_file is not None:
            metadata = Metadata()
            metadata.read(metadata_file)
            return self.process(input_file, metadata=metadata)

        # use a default .json file when neither metadata_file nor metadata were given:
        if metadata is None:
            standard_metadata_file = \
                pathlib.Path(input_file).with_suffix('.json')
            if standard_metadata_file.exists():
                Warning('Using existing standard metadata file: '
                        + str(standard_metadata_file))
                return self.process(input_file, metadata_file=standard_metadata_file)

            Warning('No metadata file provided and no default file found.')
            # in this case, metadata is left None!

        input_data = Data()
        input_data.read(input_file)

        return self.train(input_data, metadata)

    def train(self, input_data, metadata):
        """Train each model in its own pipeline using input_data and metadata,
        score, select and return the best scoring model.
        """
        input_data.set_metadata(metadata)

        pipelines = [TVAEPipeline(input_data),
                     CTGANPipeline(input_data),
                     CopulaGANPipeline(input_data),
                     GaussianCopulaPipeline(input_data)]

        self._add_scoring(ASyH.metrics.anonymity.maximum_cosine_similarity,
                          pipelines=pipelines)
        sdv_kscomplement = \
            ASyH.metrics.adapt_sdv_metric(KSComplement)
        self._add_scoring(sdv_kscomplement, pipelines=pipelines)
        sdv_cstest = \
            ASyH.metrics.adapt_sdv_metric(CSTest)
        self._add_scoring(sdv_cstest, pipelines=pipelines)

        sdv_correlation_similarity = \
            ASyH.metrics.adapt_sdv_metric_normalized(CorrelationSimilarity)
        self._add_scoring(sdv_correlation_similarity, pipelines=pipelines)

        # sdv_ML_peformance_logistic_regression = \
        #     ASyH.metrics.adapt_sdv_metric_normalized(BinaryLogisticRegression)
        # self._add_scoring(sdv_ML_peformance_logistic_regression, pipelines=pipelines)
        # ...

        self._results = concurrent_dispatch(*pipelines)

        self._best = self._select_best(self._results, pipelines=pipelines)

        return self._best
