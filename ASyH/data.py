# ASyH simple CSV and Excel file Input/Output, as well as model i/o using pkl
#
# ToDos:
#   A real exception thrown when the file type cannot be determined by read().
#
#   Dataset naming?  This could help with consistent output file naming even in
#   downstream modules like *Model.
#
#   Module docstring

import pandas
import magic
import re


# ASyH reads data into a pandas dataframe, so, the return type of the read-in
# methods are pandas dataframes.
class Data:
    # a pandas data frame holding all the actual data
    @property
    def data(self):
        return self._data

    def metadata(self):
        return self._metadata

    def __init__(self, data=None, metadata=None):
        self._data = data
        self._metadata = metadata

    def read(self, input_file): # generic read method inferring file format from 'magic'
        filetype = magic.from_file(input_file)
        x = re.compile(".*Excel.*")
        c = re.compile(".*CSV.*")
        if (x.match(filetype)):
            data = pandas.read_excel(input_file)
        elif(c.match(filetype)):
            data = pandas.read_csv(input_file)
        else:
            raise DataError("Cannot determine input file type: ") from LookupError
        # the data contains a spurious index column when saved from pandas!
        self._data = data.drop(data.columns[0], axis=1)

    def set_metadata(self, metadata=None):
        self._metadata = metadata


class RealData(Data):
    def __init__(self, data):
        Data.__init__(self, data)


# Synthetic data, in contrast to real data, can be written to files
class SyntheticData(Data):
    def __init__(self, data):
        Data.__init__(self, data)

    def write(self, outputfile):
        x = re.compile(".xls.?")
        if (x.match(outputfile) | x.match(outputfile)):
            self._data.write_excel(outputfile)
        else:
            # the default is CSV, irrespective of the file suffix.
            self._data.write_csv(outputfile)
