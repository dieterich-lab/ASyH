'''Wrapper functions for SDV metrics'''


def adapt_sdv_metric(sdv_metric_object):
    '''Return a function to apply with the metric hook of ASyH.'''
    def closure(orig_data, synth_data):
        return sdv_metric_object.compute(orig_data.data,
                                         synth_data.data,
                                         orig_data.metadata)

    return closure
