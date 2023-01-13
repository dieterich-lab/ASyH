'''Metric functions to estimate anonymity of synthetic data.'''
import math
import numpy
import pandas
import gower


def mean_pairwise_distance(orig_data, synth_data):
    '''Calculate the minimum Gower\'s distance between original and synthetic
    data.'''
    # concat orig_data with synth_data
    data_union = pandas.concat([orig_data.data, synth_data.data], axis=0)
    # calculate the Gower's distance matrix
    matrix = gower.gower_matrix(numpy.asarray(data_union))
    # extract the part in which orig_data is compared with synth_data:
    orig_size = orig_data.data.shape[0]
    synth_size = synth_data.data.shape[0]
    distances = matrix[:orig_size, synth_size:].flatten()
    # return the mean over all non-NaN values:
    return distances[~numpy.isnan(distances)].mean()


def _categorical_inner(a, b):
    '''Inner product for arrays - of same size - with categorical elements.  The
    simple product of two categorical variables x and y is 1 if x==y and 0
    otherwise.  It assumes that the arrays have been cleared of None values.
    '''
    def _categorical_mult(a_i, b_i):
        if a_i != b_i:
            return 0
        return 1

    return sum((_categorical_mult(a[i], b[i])
                for i in range(len(a))))


def _mixed_abs(numerical_array, categorical_array):
    '''Generalized vector magnitude for mixed numerical/categorical arrays.
    Provide the numerical part and the categorical part separately'''
    return \
        math.sqrt(numpy.inner(numerical_array, numerical_array) +
                  _categorical_inner(categorical_array, categorical_array))


def _mixed_cosine(numerical_array_a, categorical_array_a,
                  numerical_array_b, categorical_array_b):
    '''Calculate the combined cosine value for numerical and categorical arrays.
    It assumes that len(numerical_array_a)==len(numerical_array_b) and
    len(categorical_array_a)=len(categorical_array_b).
    '''
    # find unspecified (None) values in any of the two arrays and ignore them in
    # both:
    nan_elements_num = \
          numpy.isnan(numerical_array_a) \
        + numpy.isnan(numerical_array_b)

    def _isnone(el):
        return el==None
    nan_elements_cat = \
          _isnone(categorical_array_a) \
        + _isnone(categorical_array_b)

    a_num = numerical_array_a[~nan_elements_num]
    b_num = numerical_array_b[~nan_elements_num]
    a_cat = categorical_array_a[~nan_elements_cat]
    b_cat = categorical_array_b[~nan_elements_cat]

    numerical_prod = numpy.inner(a_num, b_num)
    categorical_prod = _categorical_inner(a_cat, b_cat)

    abs_a = _mixed_abs(a_num, a_cat)
    abs_b = _mixed_abs(b_num, b_cat)

    return (numerical_prod + categorical_prod) / (abs_a * abs_b)


def maximum_cosine_similarity(orig_data, synth_data):
    '''Find the maximum cosine value between real and synthetic data as a
    measure for maximum entry similarity.  The cosine value is calculated by
    treating the arrays in the function mixed cosine, which augments the concept
    of inner product to arrays of categorical values.
    '''
    # divide datasets into numerical and categorical tables:
    numerical_vars = orig_data.metadata().variables_by_type("numerical")
    numerical_orig = numpy.array(orig_data.data[numerical_vars])
    numerical_synt = numpy.array(synth_data.data[numerical_vars])

    categorical_vars = orig_data.metadata().variables_by_type("categorical")
    categorical_orig = numpy.array(orig_data.data[categorical_vars])
    categorical_synt = numpy.array(synth_data.data[categorical_vars])

    cosines = []
    for i in range(orig_data.data.shape[0]):
        for j in range(synth_data.data.shape[0]):
            # add cosine value for orig_data[i] and synth_data[j] to end of
            # cosines:
            cosines.append(
                _mixed_cosine(numerical_orig[i], categorical_orig[i],
                              numerical_synt[j], categorical_synt[j]))
    return max(cosines)
