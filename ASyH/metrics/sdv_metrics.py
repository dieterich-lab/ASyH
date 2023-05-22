"""Wrapper functions for SDV metrics."""
from typing import Type

from sdmetrics.single_table import SingleTableMetric
from ASyH import Data


def adapt_sdv_metric(sdv_metric_class: Type[SingleTableMetric]):
    """Wrap SDV metric classes methods .compute(), return a function to apply
    with the metric hook of ASyH.
    """
    def closure(orig_data: Data, synth_data: Data):
        return sdv_metric_class.compute(orig_data.data,
                                        synth_data.data,
                                        orig_data.metadata.metadata)
    closure.__name__ = sdv_metric_class.__name__
    return closure
