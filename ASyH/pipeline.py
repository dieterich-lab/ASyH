# ASyH Pipeline base class

from ASyH.abstract_pipeline import AbstractPipeline
import asyncio


class Pipeline(AbstractPipeline):

    def __init__(self, input_data=None, scoring_fn=None):
        self._input_data = input_data
        self._scoring_fn = scoring_fn

    async def p_run(self):
        '''Abstract method for ASyH pipeline definitions'''
        self._pipeline()

    def run(self):
        '''Start the pipeline'''
        self._pipeline()


AbstractPipeline.register(Pipeline)
