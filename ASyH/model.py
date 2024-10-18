'''ASyH Model base class'''
#
# ToDos:
#   read() and save()

from datetime import datetime
import os.path
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any, Dict

from pandas import DataFrame
from sdv.single_table.base import BaseSingleTableSynthesizer

from ASyH.data import Data


class Model(ABC):
    '''ASyH Generic Model Interface'''

    @property
    def sdv_model(self):
        return self._sdv_model

    @property
    def model_type(self):
        return self._model_type

    def __init__(
            self,
            sdv_model_class: Optional[Callable[..., BaseSingleTableSynthesizer]] = None,
            data: Optional[Data] = None,
            override_args: Optional[Dict[str, Any]] = None
    ):
        self._sdv_model = None
        self._sdv_model_class = sdv_model_class
        self._model_type = sdv_model_class.__name__
        self._override_args = override_args

        self._input_data_size = 0
        if data:
            self._training_data = data
            self._metadata = data.metadata
            self._input_data_size = data.data.shape[0]
        else:
            self._training_data = None
            self._metadata = None

        self._trained = False

    def _train(self, data: Optional[Data] = None):
        assert self._training_data is not None or data is not None
        if data is None:
            data = self._training_data
        if self._sdv_model is None:
            # create the SDV model just when we need it
            args = self.adapted_arguments(data)
            if self._override_args is not None:
                args.update(self._override_args)
            self._sdv_model = \
                self._sdv_model_class(**args)
        self._sdv_model.fit(data.data)
        self._input_data_size = data.data.shape[0]
        self._trained = True

    def save(self, filename: Optional[str] = None):
        '''Save the SDV model to pkl.'''
        if not self._sdv_model:
            return
        if not filename:  # create a filename from model type and date/time
            filename = self._model_type \
                + str(datetime.now().isoformat(timespec='auto'))
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        self._sdv_model.save(filename)

    def read(self, input_filename: str):
        '''Read the SDV model from pkl.'''
        # does filename exist?
        if not os.path.exists(input_filename):
            Warning('Model input file not found!')
            return
        if self._sdv_model:
            self._sdv_model.read(input_filename)

    def synthesize(self, sample_size: int = -1) -> DataFrame:
        '''Create synthetic data.'''
        if not self._trained:
            self._train(None)
        if sample_size == -1:
            sample_size = self._input_data_size
        return self.sdv_model.sample(sample_size)

    @abstractmethod
    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Method for specific models to adapt the constructor arguments to
        input data.  This method is supposed to be overridden by the specific
        Model to produce the correct argument dictionary {keyword: value} for
        its constructor depending on argument \'data\'.'''
        return {}


class ModelX(ABC):
    '''ASyH Extended Model Interface'''

    @property
    def ext_model(self):
        return self._sdv_model

    @property
    def model_type(self):
        return self._model_type

    def __init__(
            self,
            ext_model_class: Optional[Callable[..., BaseSingleTableSynthesizer]] = None,
            data: Optional[Data] = None,
            override_args: Optional[Dict[str, Any]] = None
    ):
        self._ext_model = None
        self._ext_model_class = sdv_model_class
        self._model_type = ext_model_class.__name__
        self._override_args = override_args

        self._input_data_size = 0
        if data:
            self._training_data = data
            self._metadata = data.metadata
            self._input_data_size = data.data.shape[0]
        else:
            self._training_data = None
            self._metadata = None

        self._trained = False

    def _train(self, data: Optional[Data] = None):
        assert self._training_data is not None or data is not None
        if data is None:
            data = self._training_data
        if self._ext_model is None:
            # create the SDV model just when we need it
            args = self.adapted_arguments(data)
            if self._override_args is not None:
                args.update(self._override_args)
            self._ext_model = \
                self._ext_model_class(**args)
        self._ext_model.fit(data.data)
        self._input_data_size = data.data.shape[0]
        self._trained = True

    def save(self, filename: Optional[str] = None):
        '''Save the ext model to pkl.'''
        if not self._ext_model:
            return
        if not filename:  # create a filename from model type and date/time
            filename = self._model_type \
                + str(datetime.now().isoformat(timespec='auto'))
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        self._ext_model.save(filename)

    def read(self, input_filename: str):
        '''Read the ext model from pkl.'''
        # does filename exist?
        if not os.path.exists(input_filename):
            Warning('Model input file not found!')
            return
        if self._ext_model:
            self._ext_model.read(input_filename)

    def synthesize(self, sample_size: int = -1) -> DataFrame:
        '''Create synthetic data.'''
        if not self._trained:
            self._train(None)
        if sample_size == -1:
            sample_size = self._input_data_size
        return self.ext_model.sample(sample_size)

    @abstractmethod
    def adapted_arguments(self, data: Optional[Data] = None) -> Dict[str, Any]:
        '''Method for specific models to adapt the constructor arguments to
        input data.  This method is supposed to be overridden by the specific
        Model to produce the correct argument dictionary {keyword: value} for
        its constructor depending on argument \'data\'.'''
        return {}
