# ASyH Model base class
#
# ToDos:
#   read() and save()

from ASyH import utils


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
        self._training_data = data.data
        self._metadata = data.metadata

    def _train(self, data=None):
        if data is None:
            self.sdv_model.fit(self._training_data)
        else:
            self.sdv_model.fit(data)
        self._trained = True

    def save(self):
        utils.ToDo()

    def read(self, input_filename):
        utils.ToDo()

    def synthesize(self, sample_size=-1):
        if not self._trained:
            self._train()
        if sample_size == -1:
            sample_size = self._input_data_size
        return self.sdv_model.sample(sample_size)
