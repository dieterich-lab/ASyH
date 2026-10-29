from os.path import dirname, join
from random import randint
import pandas
import pytest

import ASyH.metadata
import ASyH.data


INPUT_DATA_FILE = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.xlsx')
INPUT_METADATA_FILE = join(dirname(dirname(__file__)), 'examples', 'Kaggle_Sirio_Libanes-16features.json')

_input_dataframe_cache = None
_input_metadata_cache = None


@pytest.fixture
def input_dataframe():
    global _input_dataframe_cache, INPUT_DATA_FILE
    if _input_dataframe_cache is None:
        _input_dataframe_cache = pandas.read_excel(INPUT_DATA_FILE)
    yield _input_dataframe_cache


@pytest.fixture
def input_metadata():
    global _input_metadata_cache, INPUT_METADATA_FILE
    if _input_metadata_cache is None:
        _input_metadata_cache = ASyH.metadata.Metadata()
        _input_metadata_cache.read(INPUT_METADATA_FILE)
    yield _input_metadata_cache


def test_properties(input_dataframe, input_metadata):
    data_object = ASyH.data.Data(input_dataframe, metadata=input_metadata)
    assert data_object.data.equals(data_object._data)
    assert data_object.metadata == data_object._metadata


def test_construct_empty_data():
    data_object = ASyH.data.Data()
    assert data_object.data is None
    assert data_object.metadata is None


def test_construct_data_with_dataframe(input_dataframe):
    data_object = ASyH.data.Data(input_dataframe)
    assert data_object.data.equals(input_dataframe)
    assert data_object.metadata is None


def test_construct_data_with_metadata_only(input_metadata):
    data_object = ASyH.data.Data(metadata=input_metadata)
    assert data_object.data is None
    assert data_object.metadata == input_metadata


def test_construct_data_with_dataframe_and_metadata(input_dataframe, input_metadata):
    data_object = ASyH.data.Data(input_dataframe,metadata=input_metadata)
    assert data_object.data.equals(input_dataframe)
    assert data_object.metadata == input_metadata


def test_read_data(input_dataframe):
    data_object = ASyH.data.Data()
    data_object.read(INPUT_DATA_FILE)
    assert data_object.data.equals(input_dataframe)
    assert data_object.metadata is None


def test_set_metadata(input_metadata):
    data_object = ASyH.data.Data()
    assert data_object.metadata is None
    data_object.set_metadata(input_metadata)
    assert data_object.metadata == input_metadata


def test_write_synthetic_data(input_dataframe, tmp_path):
    data_object = ASyH.data.SyntheticData(input_dataframe)
    tmp_xlsx_file = str(tmp_path / 'test_write_synthetic_data.xlsx')
    data_object.write(tmp_xlsx_file)
    read_xlsx_object = ASyH.data.Data()
    read_xlsx_object.read(tmp_xlsx_file)
    assert read_xlsx_object.data.equals(input_dataframe)
    tmp_csv_file = str(tmp_path / 'test_write_synthetic_data.csv')
    data_object.write(tmp_csv_file)
    read_csv_object = ASyH.data.Data()
    read_csv_object.read(tmp_csv_file)
    # for some reason, when reading from csv, the precision of, e.g.
    # -0,00279835390946492 gets truncated (to -0,0027983539094649).  The
    # DataFrame.equals() approach suffers from this.  But
    # pandas.testing.assert_frame_equal() does not have this problem (with its
    # default arguments)
    pandas.testing.assert_frame_equal(read_csv_object.data, input_dataframe)
