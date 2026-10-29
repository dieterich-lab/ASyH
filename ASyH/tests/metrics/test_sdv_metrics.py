from sdmetrics.single_table import NewRowSynthesis
from pandas import DataFrame

from ASyH import Metadata
from ASyH.metrics.sdv_metrics import adapt_sdv_metric
from ASyH.data import Data


EXAMPLE_METADATA = Metadata(metadata={
    'columns': {
        'name': {'sdtype': 'id'},
        'sex': {'sdtype': 'categorical'},
        'height': {'sdtype': 'numerical'},
    },
})

EXAMPLE_DATA_A = Data(
    data=DataFrame(data={
        'name': ['Albert', 'Berta', 'Charlie', 'Dorothea'],
        'sex':  ['m', 'f', 'm', 'f'],
        'height': [1.80, 1.60, 1.75, 1.78],
    }),
    metadata=EXAMPLE_METADATA
)

EXAMPLE_DATA_B = Data(
    data=DataFrame(data={
        'name': ['Emil', 'Frida'],
        'sex':  ['m', 'f'],
        'height': [1.80, 1.30],
    }),
    metadata=EXAMPLE_METADATA
)


def test_adapt_sdv_metric():
    metric = adapt_sdv_metric(NewRowSynthesis)
    assert metric.__name__ == 'NewRowSynthesis'
    assert metric(EXAMPLE_DATA_A, EXAMPLE_DATA_B) == 0.5
    assert metric(EXAMPLE_DATA_A, EXAMPLE_DATA_A) == 0
