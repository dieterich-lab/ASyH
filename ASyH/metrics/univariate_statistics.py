import scipy.stats
import collections
import numpy
import sdv.metrics


def kstest(real_data, synthetic_data):
    '''Calculate p-values for all numerical variables in input data.'''
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

    def frequencies_ordered(data, var):
        freqs = dict(collections.Counter(numpy.array(data.data.loc[:, var])))
        return [freqs[cat] for cat in sorted(freqs.keys())]

    def average(value_list):
        return sum(value_list)/len(value_list)

    results = {}
    for var in categorical_variables:
        real_freqs = frequencies_ordered(real_data, var)
        synth_freqs = frequencies_ordered(synthetic_data, var)
        results[var] = \
            scipy.stats.chisquare(real_freqs, synth_freqs).pvalue

    results['average'] = average(results.values())
    return results


def cstest_sdv(real_data, synthetic_data):
    '''Chi-square test as defined in SDV.'''
    test = sdv.metrics.tabular.CSTest()
    return test.compute(real_data.data, synthetic_data.data)
