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
                                        orig_data.metadata.columns)
    closure.__name__ = sdv_metric_class.__name__
    return closure


def adapt_sdv_metric_normalized(sdv_metric_class: Type[SingleTableMetric]):
    """Wrap SDV metric classes methods .compute() and .normalize(), returning a
    function to apply with the metric hook of ASyH.
    """
    def closure(orig_data: Data, synth_data: Data):
        raw_score = sdv_metric_class.compute(orig_data.data,
                                             synth_data.data,
                                             orig_data.metadata.columns)
        return sdv_metric_class.normalize(raw_score)
    closure.__name__ = sdv_metric_class.__name__
    return closure
