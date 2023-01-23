# ASyH's simple metadata inference

import pdb

import pathlib
import json
from typing import Optional, Union, Dict, Any

from pandas import DataFrame

from ASyH.dataerror import DataError


class Metadata:
    'A wrapper class for SDV metadata.'

    def _skeleton(self, data):
        return

    def _create_skeleton_file(self, data):
        return

    @property
    def fields(self):
        'Property method for retrieving the \'fields\' entry in the metadata.'
        if self._get_set_tablename() is None:
            return None
        return self.metadata['tables'][self._tablename]['fields']

    @property
    def table(self):
        'Property method for retrieving the (zeroth) \'tables\' entry in the metadata.'
        if self._get_set_tablename() is not None:
            return self.metadata['tables'][self._tablename]
        return None

    def read(self, filename: Union[str, pathlib.Path]):
        '''Read the metadata from file into the metadata dict.'''
        if pathlib.Path(filename).is_file():
            with open(filename, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError
        self._validate_metadata(self.metadata)
        self._tablename = self._get_set_tablename()

    def save(self, out_filename: str):
        '''Save the metadata dict to json file.'''
        # (later: +annotating the possible SDV type)
        with open(out_filename, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)

    def variables_by_type(self, type_string):
        '''Filter the dataset's variables by the string in their type
        information.'''
        assert self._tablename is not None
        field_types = self.fields
        return [key  # the key, i.e. variable name
                for key, typeinfo in field_types.items()
                if typeinfo['type'] == type_string]

    def __init__(self, data: Optional[Union[DataFrame, Dict[str, Any]]] = None):
        # for now just use the dataFrame's dtypes:
        if data is not None:
            self.metadata = data.dtypes.to_dict()
            self._validate_metadata(self.metadata)
            self._tablename = self._get_set_tablename()
        else:
            self.metadata = None
            self._tablename = None

    def _infer(self, data_column):
        # ToDo!
        # for now, just return the dtype
        return data_column.dtype

    def _infer_metadata(self, data_frame):
        meta = {x: self._infer(data_frame[x]) for x in data_frame.columns}
        return meta

    def _validate_metadata(self, metadata: Dict[str, Any]):
        if 'tables' not in metadata.keys():
            raise DataError('Metadata malformed: no \'tables\' entry in outermost scope.')

        if len(metadata['tables']) > 1:
            Warning('Metadate describes a multitable dataset. ASyH works only \
            with single table datasets.\n Using table' + list(metadata['tables'].keys())[0])

    def _get_set_tablename(self):
        if self._tablename is not None:
            return self._tablename
        if self.metadata is None:
            return None
        self._validate_metadata(self.metadata)
        self._tablename = list(self.metadata['tables'].keys())[0]
        return self._tablename
