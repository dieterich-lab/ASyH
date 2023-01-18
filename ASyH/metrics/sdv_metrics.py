"""Wrapper functions for SDV metrics."""


def adapt_sdv_metric(sdv_metric_class):
    """Wrap SDV metric classes methods .compute(), return a function to apply
    with the metric hook of ASyH.
    """
    def closure(orig_data, synth_data):
        return sdv_metric_class.compute(orig_data.data,
                                        synth_data.data,
                                        orig_data.metadata.metadata)
    closure.__name__ = sdv_metric_class.__name__
    return closure


def adapt_sdv_metric_normalized(sdv_metric_class):
    """Wrap SDV metric classes methods .compute() and .normalize(), returning a
    function to apply with the metric hook of ASyH.
    """
    def closure(orig_data, synth_data):
        raw_score = sdv_metric_class.compute(orig_data.data,
                                             synth_data.data,
                                             orig_data.metadata.metadata)
        return sdv_metric_class.normalize(raw_score)
    closure.__name__ = sdv_metric_class.__name__
    return closure
