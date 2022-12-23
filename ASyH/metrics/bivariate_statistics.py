import math
import scipy.stats
import numpy
import sklearn.decomposition


# pairwise application of bivariate function:
def pairwise(func, dataframe, list_of_variables=None):
    '''Apply func to any pair which can be combined from list_of_variables. If
    list_of_variables is not specified, all columns of dataframe are used.  It
    is assumed that func(a, b) = func(b, a).    '''
    returnval = {}
    if not list_of_variables:
        list_of_variables = dataframe.columns
    for i, first_var in enumerate(list_of_variables):
        first_data = numpy.array(dataframe.loc[:, first_var])
        for second_var in list_of_variables[i+1:]:
            second_data = numpy.array(dataframe.loc[:, second_var])
            returnval[first_var, second_var] = func(first_data, second_data)
    return returnval


def comparison(real_data, synthetic_data, calculating_fn, comparison_fn):
    '''Template function for comparing bivariate functions applied on each pair
    of numerical variables in the dataframes of real_data and synthetic_data.
    Arguments:
      real_data      - ASyH.data.Data object holding the real data.
      synthetic_data - ASyH.data.Data object holding the synthetic data.
      calculating_fn - bivariate function to be applied to each pair of columns
                       in the given ASyH.data.Data containers.
      comparison_fn  - function to calculate a single value from one of the
                       results of calculating_fn on real data and the
                       corresponding result of calculating_fn on synthetic data.
    '''
    returnval = {}
    variables = real_data.metadata().variables_by_type("numerical")
    if len(variables) > 1:
        real_vals = pairwise(calculating_fn, real_data.data, variables)
        synth_vals = pairwise(calculating_fn, synthetic_data.data, variables)
        for i in real_vals.keys():
            returnval[i] = comparison_fn(real_vals[i], synth_vals[i])
    return returnval


def cosine_comparator(vec_a, vec_b):
    '''Calculate cosine between the two vectors vec_a and vec_b.'''
    scalar_product = vec_a @ vec_b
    abs_a = numpy.linalg.norm(vec_a)
    abs_b = numpy.linalg.norm(vec_b)
    return scalar_product / (abs_a*abs_b)


def principal_components(column_a, column_b):
    '''Bivariate principal component calculation using scikit-learn.'''
    data = [[column_a[i], column_b[i]] for i in range(len(column_a))]
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(data)
    return pca.components_[0]


def pc_comparison(real_data, synthetic_data):
    '''Calculate cosines of first principal components of each pair of numerical
    values in real_data and synthetic_data.'''
    return comparison(real_data,
                      synthetic_data, principal_components, cosine_comparator)


def pearson_correlation(column_a, column_b):
    '''Calculate Pearson\'s correlation between numpy arrays column_a and
    column_b.'''
    (r, p) = scipy.stats.pearsonr(column_a, column_b)
    if p > 0.05:
        return 0
    return r


def pearsonr_comparison(real_data, synthetic_data):
    '''Calculate diffences in Pearson\'s Correlation of each pair of numerical
    values in real_data and synthetic_data.'''
    return comparison(real_data, synthetic_data,
                      pearson_correlation,
                      lambda x, y: math.fabs(x-y)/(1, x)[x != 0])


def spearman_correlation(column_a, column_b):
    '''Calculate Spearman\'s Correlation between numpy arrays column_a and
    column_b.'''
    (r, p) = scipy.stats.spearmanr(column_a, column_b)
    if p > 0.05:
        return 0
    return r


def spearmanr_comparison(real_data, synthetic_data):
    '''Calculate differences in Spearman\'s correlation of each pair of
    numerical values in real_data and synthetic_data'''
    return comparison(real_data, synthetic_data,
                      spearman_correlation,
                      lambda x, y: math.fabs(x-y)/(1, x)[x != 0])
