# ASyH simple CSV and Excel file Input/Output, as well as model i/o using pkl
#
# ToDos:
#   A real exception thrown when the file type cannot be determined by read().
#
#   Dataset naming?  This could help with consistent output file naming even in
#   downstream modules like *Model.
#
#   Module docstring
from typing import Optional
import re
import magic

import pandas
from pandas import DataFrame

from ASyH.metadata import Metadata

from ASyH.dataerror import DataError


# ASyH reads data into a pandas dataframe, so, the return type of the read-in
# methods are pandas dataframes.
class Data:
    # a pandas data frame holding all the actual data
    @property
    def data(self) -> pandas.DataFrame:
        return self._data

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    def __init__(self, data: Optional[DataFrame] = None, metadata: Optional[Metadata] = None):
        self._data = data
        self._metadata = metadata

    def read(self, input_file):
        """Generic read method inferring file format from \'magic\'.  Excel and
        CSV are supported.
        """
        filetype = magic.from_file(input_file)
        x = re.compile(".*Excel.*")
        c = re.compile(".*CSV.*")
        if x.match(filetype):
            # rounding the input to work around
            # https://github.com/sdv-dev/SDV/issues/1039
            data = pandas.read_excel(input_file).round(decimals=14)
        elif c.match(filetype):
            data = pandas.read_csv(input_file).round(decimals=14)
        else:
            raise DataError("Cannot determine input file type: ")
        # the data contains a spurious index column when saved from pandas!
        self._data = data.drop(data.columns[0], axis=1)

    def set_metadata(self, metadata: Optional[Metadata] = None):
        self._metadata = metadata


class RealData(Data):
    def __init__(self, data: Optional[DataFrame]):
        Data.__init__(self, data)


class SyntheticData(Data):
    """Synthetic data class: In contrast to real data, it can be written to
    file.
    """

    def __init__(self, data: Optional[DataFrame]):
        Data.__init__(self, data)

    def write(self, outputfile):
        """Write synthetic data to output to file.  Excel and CSV are supported.
        The default is CSV, if the given output file name isn't compatible with
        Excel file endings.
        """
        x = re.compile(".xls.?")
        if x.search(outputfile):
            self._data.to_excel(outputfile)
        else:
            self._data.to_csv(outputfile)
