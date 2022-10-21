from abc import ABC, abstractmethod


class AbstractPipeline(ABC):
    '''Abstract class for ASyH pipelines.'''

    @abstractmethod
    def _pipeline(self):
        pass  # To be overwritten by the derived pipeline class
