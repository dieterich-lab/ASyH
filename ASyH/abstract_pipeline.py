from abc import ABC, abstractmethod


class AbstractPipeline(ABC):
    '''Abstract class for ASyH pipelines.'''

    @abstractmethod
    def run(self):
        pass
