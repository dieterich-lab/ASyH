# ASyH's simple metadata inference

import pathlib
import json
from typing import Optional, Union, Dict, Any

from pandas import DataFrame
from sdv.metadata.single_table import SingleTableMetadata


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
        # If both metadata and data are specified, metadata is used.
        the_metadata = metadata
        if metadata is None and data is not None:
            dummy = SingleTableMetadata()
            dummy.detect_from_dataframe(data)
            the_metadata = dummy.to_dict()
        self.metadata = the_metadata
