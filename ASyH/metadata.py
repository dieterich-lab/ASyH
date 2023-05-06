# ASyH's simple metadata inference

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
    def columns(self):
        'Property method for retrieving the \'columns\' entry in the metadata.'
        if self.metadata is None:
            return {}
        return self.metadata['columns']

    def read(self, filename: Union[str, pathlib.Path]):
        '''Read the metadata from file into the metadata dict.'''
        if pathlib.Path(filename).is_file():
            with open(filename, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            raise FileNotFoundError

    def save(self, out_filename: str):
        '''Save the metadata dict to json file.'''
        # (later: +annotating the possible SDV type)
        with open(out_filename, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f)

    def variables_by_type(self, type_string):
        '''Filter the dataset's variables by the string in their type
        information.'''
        columns = self.columns
        return [key  # the key, i.e. variable name
                for key, typeinfo in columns.items()
                if typeinfo['sdtype'] == type_string]

    def __init__(self,
                 metadata: Optional[Dict[str, Any]] = None,
                 data: Optional[DataFrame] = None):
        # ToDo: (branch sdv-1.0.0) use autodetect!
        # If both metadata and data are specified, metadata is used.

        if metadata is None and data is not None:
            metadata = {'columns': data.dtypes.to_dict()}
        self.metadata = metadata

    def _infer(self, data_column):
        # ToDo: as of sdv 1.0.0 sdv.metadata.SingleTableMetadata has
        # a method named detect_from_dataframe(.)
        # for now, just return the dtype
        return data_column.dtype

    def _infer_metadata(self, data_frame):
        meta = {x: self._infer(data_frame[x]) for x in data_frame.columns}
        return meta
