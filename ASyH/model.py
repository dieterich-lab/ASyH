# ASyH Model base class
#
# ToDos:
#   read() and save()

from datetime import datetime
import os.path
# from ASyH import utils


class Model:

    @property
    def sdv_model(self):
        return self._sdv_model

    @property
    def model_type(self):
        return self._model_type

    _input_data_size = 0
    _trained = False

    def __init__(self, model_type=None, sdv_model=None, data=None):
        self._model_type = model_type
        self._sdv_model = sdv_model
        if data:
            self._training_data = data.data
            self._metadata = data.metadata
            self._input_data_size = data.data.shape[0]
        else:
            self._training_data = None
            self._metadata = None

    def _train(self, data=None):
        assert(self._training_data is not None or data is not None)
        if data is None:
            self.sdv_model.fit(self._training_data)
        else:
            self.sdv_model.fit(data)
            self._input_data_size = data.shape[0]
        self._trained = True

    def save(self, filename=None):
        '''Save the SDV model to pkl.'''
        if not self._sdv_model:
            return
        if not filename:  # create a filename from model type and date/time
            filename = self._model_type \
                + str(datetime.now().isoformat(timespec='auto'))
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        self._sdv_model.save(filename)

    def read(self, input_filename):
        '''Read the SDV model from pkl.'''
        # does filename exist?
        if not os.path.exists(input_filename):
            Warning('Model input file not found!')
            return
        if self._sdv_model:
            self._sdv_model.read(input_filename)

    def synthesize(self, sample_size=-1):
        if not self._trained:
            self._train()
        if sample_size == -1:
            sample_size = self._input_data_size
        return self.sdv_model.sample(sample_size)
