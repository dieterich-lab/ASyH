import scipy.stats
import collections
import numpy
import sdv.metrics


def kstest(real_data, synthetic_data):
    '''Calculate p-values for all numerical variables in input data.'''
    # for each continuous variable:
    numerical_variables = \
        real_data.metadata.variables_by_type("numerical")
    return [
        scipy.stats.kstest(
            numpy.array(real_data.data.loc[:, col].values),
            numpy.array(synthetic_data.data.loc[:, col].values)
        ).pvalue
        for col in numerical_variables]


def cstest(real_data, synthetic_data):
    '''Chi-square test metric.'''
    categorical_variables = \
        real_data.metadata.variables_by_type("categorical")

    r_frequencies = {
        var:
        dict(collections.Counter(numpy.array(real_data.data.loc[:, var])))
        for var in categorical_variables}
    s_frequencies = {
        var:
        dict(collections.Counter(numpy.array(synthetic_data.loc[:, var])))
        for var in categorical_variables}

    results = {}
    for var in categorical_variables:
        r_freq_dict = r_frequencies[var]
        s_freq_dict = s_frequencies[var]

        r_freqs_serial = [r_freq_dict[cat]
                          for cat in sorted(r_freq_dict.keys())]
        s_freqs_serial = [s_freq_dict[cat]
                          for cat in sorted(s_freq_dict.keys())]

        results[var] = \
            scipy.stats.chisquare(r_freqs_serial, s_freqs_serial).pvalue

    results_values = results.values()
    avg = sum(results_values) / len(results_values)
    results['average'] = avg
    return results


def cstest_sdv(real_data, synthetic_data):
    '''Chi-square test as defined in SDV.'''
    test = sdv.metrics.tabular.CSTest()
    return test.compute(real_data.data, synthetic_data.data)
