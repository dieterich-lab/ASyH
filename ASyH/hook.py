'''Hooks: simple execution hooks for common point of execution.'''


class Hook:
    '''A simple class for execution \'hooks\', i.e. a variable to collect
    functions which can be executed all in the order they were specified at a
    chosen point in the program.'''

    _function_list = []

    def add(self, func):
        '''Add a function to the hook'''
        self._function_list.append(func)

    def execute(self, *args):
        '''Return a dict of the format function: return_value.'''
        return {func.__name__: func(*args)
                for func in self._function_list}


class ScoringHook(Hook):
    '''Hook for Scoring functions, i.e. with the fingerprint
    (real_data, synthetic_data) => float,
    where real_data and synthetic_data are objects of the ASyH.data.Data
    class.  Only add such functions, otherwise exceptions will be thrown.'''

    def execute(self, real_data, synthetic_data):
        '''Execute all scoring functions in the hook,
        return a dict of the format function: return_value.'''
        return {func.__name__: func(real_data, synthetic_data)
                for func in self._function_list}
