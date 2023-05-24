import re
from os.path import join

import pytest
from pandas import DataFrame
from ASyH.report import Report, ImagesDict
from io import StringIO, BytesIO

DEFAULT_SCORE = 0.31314
DEFAULT_COLUMN_SHAPE_SCORE = 0.2121
DEFAULT_COLUMN_PAIR_TRENDS_SCORE = 0.57121
DEFAULT_INPUT_DATA = '__INPUT__'
DEFAULT_SYNTHETIC_DATA = '__SYNTHETIC__'
DEFAULT_META_DATA = {
    'columns': {
        'id_col': {'sdtype': 'id'},
        'data_1': {'sdtype': 'categorical'},
        'data_2': {'sdtype': 'numerical'},
    }
}

DEFAULT_IMAGES = ImagesDict(
    column_shapes='col_shapes.png',
    column_pair_trends='col_pair_trends.png',
    per_column=['data_1.png', 'data_2.png']
)

DEFFAULT_IMAGE_DIR = 'img_dir'


class FigureMock:
    def __init__(self, what):
        self._what = what

    def write_image(self, file, format):  # noqa
        if format != 'png':
            raise "We only do .png!"
        file.write(f"PNG of {self._what}".encode('utf-8'))


class SDMetricsReportMock:
    def generate(self, raw_data, synthetic_data, metadata):  # noqa
        pass

    def get_details(self, property_name):  # noqa
        return DataFrame(
            data={
                'Column': ['Mock'],
                'Metric': ['Mock'],
                'Quality Score': [1.0],
            }
        )

    def get_score(self):  # noqa
        return DEFAULT_SCORE

    def get_properties(self):  # noqa
        return DataFrame(
            data={
                'Property': ['Column Shapes', 'Column Pair Trends'],
                'Score': [DEFAULT_COLUMN_SHAPE_SCORE, DEFAULT_COLUMN_PAIR_TRENDS_SCORE]
            }
        )

    def get_visualization(self, property_name):  # noqa
        return FigureMock(property_name)


@pytest.fixture
def pickle_mock(mocker):
    def my_mock(obj, file_like):
        file_like.write(f"Pickled {obj.__class__.__name__}".encode('utf-8'))

    mocker.patch('ASyH.report.pickle.dump', new=my_mock)


@pytest.fixture
def default_report():
    report = Report(
        input_data=DEFAULT_INPUT_DATA,
        synthetic_data=DEFAULT_SYNTHETIC_DATA,
        metadata=DEFAULT_META_DATA,
        sdmetrics_report=SDMetricsReportMock(),
    )
    report._image_dir = DEFFAULT_IMAGE_DIR
    yield report

@pytest.fixture
def mock_sdmetrics(mocker):
    def my_mock(real_data, synthetic_data, column_name, metadata):  # noqa
        return FigureMock(column_name)

    mocker.patch('ASyH.report.sdmetrics.reports.utils.get_column_plot', new=my_mock)


def test_create_pickled_report(pickle_mock, default_report):
    result = BytesIO()
    default_report.create_pickled_report(result)
    assert result.getvalue() == b'Pickled SDMetricsReportMock'


def test_create_scores_csv(default_report):
    result = StringIO()
    default_report.create_scores_csv(result)
    assert result.getvalue() == ''',Column,Metric,Quality Score
0,Mock,Mock,1.0
'''


def test_get_columns(default_report):
    assert default_report.get_columns() == ['data_1', 'data_2']


def test_get_mark_down_report(default_report):
    result = default_report.get_mark_down_report('test_data_set', 'test_model', DEFAULT_IMAGES)
    assert f"QualityScore: {100*DEFAULT_SCORE:.2f}%" in result
    assert f"| {100*DEFAULT_COLUMN_SHAPE_SCORE:.2f} % | {100*DEFAULT_COLUMN_PAIR_TRENDS_SCORE:.2f} % |" in result
    for image in DEFAULT_IMAGES.per_column:
        assert f"![]({ image })" in result


def test_get_report_property_as_percent(default_report):
    assert default_report.get_report_property_as_percent('Column Shapes') == 100 * DEFAULT_COLUMN_SHAPE_SCORE


def test_create_per_column_image(default_report, mock_sdmetrics):
    result = BytesIO()
    default_report.create_per_column_image('data_1', result)
    assert result.getvalue() == b'PNG of data_1'


@pytest.mark.parametrize(
    'column,file_name',
    [
        ('bla', 'column_plot_bla.png'),
        ('foo mol/l', 'column_plot_foo_molXl.png'),
        ('Müller-Faktor', 'column_plot_MXller-Faktor.png'),
    ]
)
def test_get_per_column_image_name(default_report, column, file_name):
    assert default_report.get_per_column_image_name(column) == join(DEFFAULT_IMAGE_DIR, file_name)


def test_get_stat_image_name(default_report):
    assert default_report.get_stat_image_name('Column Pair Trends') \
           == join(DEFFAULT_IMAGE_DIR, 'column_pair_trends.png')


def test_create_stats_image(default_report, mock_sdmetrics):
    result = BytesIO()
    default_report.create_stats_image('Column Pair Trends', result)
    assert result.getvalue() == b'PNG of Column Pair Trends'
