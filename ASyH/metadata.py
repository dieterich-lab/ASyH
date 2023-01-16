# ASyH's simple metadata inference

import pathlib
# import utils
import json


class Metadata:

    # metadata generally is a dict:
    metadata = {}

    def _skeleton(self, data):
        return

    def _create_skeleton_file(self, data):
        return

    def read(self, filename):
        '''Read the metadata from file into the metadata dict.'''
        if pathlib.Path(filename).is_file():
            with open(filename, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError

    def save(self, outfilename):
        '''Save the metadata dict to json file.'''
        # (later: +annotating the possible SDV type)
        with open(outfilename, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)

    def variables_by_type(self, type_string):
        '''Filter the dataset's variables by the string in their type
        information.'''
        field_types = self.metadata['tables']['data']['fields']
        return [key  # the key, i.e. variable name
                for key, typeinfo in field_types.items()
                if typeinfo['type'] == type_string]

    def __init__(self, data=None):
        # for now just use the dataFrame's dtypes:
        if data is not None:
            self.metadata = data.dtypes.to_dict()

    def _infer(self, data_column):
        # ToDo!
        # for now, just return the dtype
        return data_column.dtype

    def _infer_metadata(self, data_frame):
        meta = {x: self._infer(data_frame[x]) for x in data_frame.columns}
        return meta
