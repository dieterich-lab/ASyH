from os.path import dirname, join
from random import randint
import pytest

import ASyH.metadata
import ASyH.data


_input_metadata_cache = None
_input_data_cache = None
_input_dict = {'columns': {'col0': {'sdtype': 'id', 'regex_format': '[0-9]'},
                           'col1': {'sdtype': 'categorical'},
                           'col2': {'sdtype': 'numerical',
                                    'computer_representation': 'Float'}
                           }
               }


@pytest.fixture
def input_metadata():
    global _input_metadata_cache
    if _input_metadata_cache is None:
        input_file = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.json')
        _input_metadata_cache = ASyH.metadata.Metadata()
        _input_metadata_cache.read(input_file)
    yield _input_metadata_cache


@pytest.fixture
def input_dict():
    global _input_dict
    yield _input_dict


@pytest.fixture
def input_data():
    global _input_data_cache
    if _input_data_cache is None:
        input_file = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.xlsx')
        _input_data_cache = ASyH.data.Data()
        _input_data_cache.read(input_file)
        metadata_file = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.json')
        _metadata = ASyH.metadata.Metadata()
        _metadata.read(metadata_file)
        _input_data_cache.set_metadata(_metadata)
    yield _input_data_cache


def test_construct_empty_metadata():
    m = ASyH.metadata.Metadata()
    assert m.metadata is None
    # .columns should just return empty dictionary
    assert m.columns == {}


def test_construct_from_dict(input_dict):
    # ToDo
    m = ASyH.metadata.Metadata(metadata=input_dict)
    mkeys = list(m.columns.keys())
    assert mkeys == ['col0', 'col1', 'col2']


def test_construct_from_data(input_data):
    # ToDo
    m = ASyH.metadata.Metadata(data=input_data.data)
    assert m.metadata is not None
    for key in m.columns.keys():
        assert m.columns[key]['sdtype'] in ['boolean', 'categorical', 'datetime', 'id', 'numerical']


def test_construct_with_data_and_dict(input_data, input_dict):
    m = ASyH.metadata.Metadata(metadata=input_dict, data=input_data.data)
    # input_dict has the higher priority!
    assert m.metadata == input_dict


def test_columns_member(input_metadata):
    assert input_metadata.columns is not None
    assert input_metadata.columns == input_metadata.metadata['columns']


def test_save(input_metadata, tmp_path):
    # use fixture, save to temporary file, read and compare with fixture.
    tempfile = tmp_path / f'tempfile-{randint(1000000,9999999)}.json'
    input_metadata.save(tempfile)
    test_metadata = ASyH.metadata.Metadata()
    test_metadata.read(tempfile)
    assert test_metadata.columns == input_metadata.columns
    assert test_metadata.metadata['columns'] == input_metadata.metadata['columns']


def test_variables_by_type(input_metadata):
    # ToDo
    case_numeric = input_metadata.variables_by_type('numerical')
    assert case_numeric == ['BLOODPRESSURE_DIASTOLIC_MEAN',
                            'BLOODPRESSURE_SISTOLIC_MEAN']
    case_categ = input_metadata.variables_by_type('categorical')
    assert case_categ == ['AGE_ABOVE65',
                          'AGE_PERCENTIL',
                          'GENDER',
                          'DISEASE GROUPING 1',
                          'DISEASE GROUPING 2',
                          'DISEASE GROUPING 3',
                          'DISEASE GROUPING 4',
                          'DISEASE GROUPING 5',
                          'DISEASE GROUPING 6',
                          'HTN',
                          'IMMUNOCOMPROMISED',
                          'OTHER',
                          'ICU']
    case_id = input_metadata.variables_by_type('id')
    assert case_id == ['PATIENT_VISIT_IDENTIFIER']
    case_boolean = input_metadata.variables_by_type('boolean')
    assert case_boolean == []
    case_datetime = input_metadata.variables_by_type('datetime')
    assert case_datetime == []
