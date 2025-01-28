'''Define the standard application of ASyH.'''
import pathlib

import sdmetrics.reports.single_table
# from sdv.metrics.tabular import KSComplement, CSTest, CorrelationSimilarity

from ASyH.data import Data
from ASyH.metadata import Metadata
from ASyH.pipelines \
    import CopulaGANPipeline, TVAEPipeline, \
    CTGANPipeline, GaussianCopulaPipeline
from ASyH.dispatch import concurrent_dispatch

from fancyimpute import IterativeImputer

# import ASyH.metrics.anonymity
# import ASyH.metrics


class Application:
    '''The standard application of ASyH.'''

    @property
    def model(self):
        return self._best

    def __init__(self, preprocess=False):
        self._results = []
        self._best = None
        self._preprocess = preprocess

    def _add_scoring(self, scoring_function, pipelines=None):
        '''Add a scoring function to all pipelines.'''
        if pipelines is None:
            pass
        for pipeline in pipelines:
            pipeline.add_scoring(scoring_function)

    def _add_preprocessing(self, preprocess_function, pipelines=None):
        '''Add a preprocessing function to the pipeline.'''
        if pipelines is None:
            pass
        for pipeline in pipelines:
            pipeline.add_preprocessing(preprocess_function)

    def _add_postprocessing(self, postprocess_function, pipeline=None):
        '''Add a postprocessing function to the pipeline.'''
        if pipelines is None:
            pass
        for pipeline in pipelines:
            pipeline.add_postprocessing(postprocess_function)

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

        def preprocess_impute(input_data):
            MICE = IterativeImputer(verbose=False)
            input_data_2 = MICE.fit_transform(input_data)
            return input_data_2
        
        def postprocess_function(synth_data):
            pass
            ...
            return post_data

        def sdmetrics_quality(input_data, synth_data):
            report = sdmetrics.reports.single_table.QualityReport()
            report.generate(input_data.data,
                            synth_data.data,
                            input_data.metadata.metadata,
                            verbose=False)
            return report.get_score()

        self._add_preprocessing(preprocess_impute, pipeline=pipelines)

        self._add_postprocessing(postprocess_function, pipelines)

        self._add_scoring(sdmetrics_quality, pipelines=pipelines)

        self._results = concurrent_dispatch(*pipelines)

        self._best = self._select_best(self._results, pipelines=pipelines)

        return self._best