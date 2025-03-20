'''Define the standard application of ASyH.'''
import pathlib

import sdmetrics.reports.single_table
# from sdv.metrics.tabular import KSComplement, CSTest, CorrelationSimilarity

from ASyH.data import Data
from ASyH.metadata import Metadata
from ASyH.pipelines \
    import CopulaGANPipeline, TVAEPipeline, \
    CTGANPipeline, GaussianCopulaPipeline, CTABGANPipeline
from ASyH.dispatch import concurrent_dispatch

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# import pudb
# from pudb.remote import set_trace
# import ASyH.metrics.anonymity
# import ASyH.metrics


class Application:
    '''The standard application of ASyH.'''

    @property
    def model(self):
        return self._best
    

    @property
    def results(self):
        return self._results
    
    def model2pipeline(self, model):
        map_model2pipeline = {
            'TVAE': TVAEPipeline,
            'CTGAN': CTGANPipeline,
            'CopulaGAN': CopulaGANPipeline,
            'GaussianCopula': GaussianCopulaPipeline,
            'CTABGAN': CTABGANPipeline
        }
        return map_model2pipeline[model]


    def __init__(self, preprocess=False, models=None):
        self._results = []
        self._best = None
        self._preprocess = preprocess
        self.models = models


    def _add_scoring(self, scoring_function, pipelines=None):
        '''Add a scoring function to all pipelines.'''
        if pipelines is None:
            pass
        for pipeline in pipelines:
            pipeline.add_scoring(scoring_function)


    def _add_preprocessing(self, preprocess_function, pipelines=None):
        '''Add a preprocessing function to the pipeline.'''
        if self._preprocess == False or pipelines is None:
            pass
        for pipeline in pipelines:
            pipeline.add_preprocessing(preprocess_function)


    def _add_postprocessing(self, postprocess_function, pipelines=None):
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

        if self.models is not None:
            pipelines = [self.model2pipeline(model)(input_data) for model in self.models]
        else:
            pipelines = [TVAEPipeline(input_data),
                        CTGANPipeline(input_data),
                        CopulaGANPipeline(input_data),
                        GaussianCopulaPipeline(input_data)]

        print(f"Running pipelines: {pipelines} ...")

        def preprocess_impute(input_data):
            # MICE = IterativeImputer(verbose=False)
            imputer = IterativeImputer(
                # estimator='RandomForestRegressor',              # Default: BayesianRidge
                estimator=None,
                max_iter=10,                 # Maximum iterations per feature
                tol=0.001,                   # Stopping tolerance threshold
                random_state=42,             # Reproducibility                                                                                     
                initial_strategy='median'      # Initial imputation method
                )
            # set_trace() # breakpoint
            imp_data = imputer.fit_transform(input_data)
            return imp_data
        
        # TODO: Implement postprocessing function
        # def postprocess_function(synth_data):
        #     pass
        #     ...
        #     return post_data

        def sdmetrics_quality(input_data, synth_data):
            report = sdmetrics.reports.single_table.QualityReport()
            report.generate(input_data.data,
                            synth_data.data,
                            input_data.metadata.metadata,
                            verbose=False)
            return report.get_score()

        self._add_preprocessing(preprocess_impute, pipelines=pipelines)
        print("Added preprocessing hooks")

        # self._add_postprocessing(postprocess_function, pipelines)

        self._add_scoring(sdmetrics_quality, pipelines=pipelines)
        print("Added scoring hooks")

        print("Dispatching the pipelines concurrently ...")
        self._results = concurrent_dispatch(*pipelines)

        self._best = self._select_best(self._results, pipelines=pipelines)

        return self._best
