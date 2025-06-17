import inspect
from typing import Dict, Any
import pandas as pd
from ASyH.data import Data
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def ToDo():
    print(inspect.currentframe().f_back.f_code.co_name +
          "This feature is not implemented yet.", end='\n')


def flatten_dict(indict: Dict[str, Any],
                 root_key: str = '',
                 key_separator: str = '.'):
    items = []
    for key, val in indict.items():
        context = root_key + key_separator + key if root_key else key
        if isinstance(val, dict):
            items.extend(
                flatten_dict(val, context, key_separator=key_separator).items())
        else:
            items.append((context, val))
    return dict(items)


# ASyH static class for preprocessing functions
class Utils:

    def __new__(cls, *args, **kwargs):
        raise TypeError("Preprocess class cannot be instantiated.")
    
    @staticmethod
    def normalize(data):
        '''Normalize the data.'''
        return data
    
    # the method that identifies the date columns in the dataframe (entry like '2021-01-01')
    # and returns the list of these columns
    @staticmethod
    def identify_date_cols(data) -> list:
        metadata = data.metadata
        date_cols = [col for col in data.data.columns if metadata.columns[col]['sdtype'] == 'datetime']
        return date_cols
    
    # the method that converts the dates in the dataframe to the number of days since the 1900-01-01
    @staticmethod
    def convert_dates(data, date_cols) -> pd.DataFrame:
        df = data.data
        for col in date_cols:
            # TODO: make sure that replace date is median value of a column
            replace_date = "1970-01-01"
            df[col] = pd.to_datetime(df[col])
            # replace nans with default date - 1970-01-01
            df[col].fillna(pd.Timestamp(replace_date), inplace=True)
            df[col] = (df[col] - pd.Timestamp("1900-01-01")) // pd.Timedelta('1D')
            # BREAK:
            # import pudb.remote as remote; remote.set_trace()
        return df
    
    # # the method that does inverse operation to convert_dates
    # @staticmethod
    # def convert_back_dates(df, date_cols) -> pd.DataFrame:

    #     assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
    #     assert isinstance(date_cols, list), "date_cols must be a list"
    #     assert all(col in df.columns for col in date_cols), "All columns in date_cols must be in df.columns"

    #     for col in date_cols:
    #         df[col] = pd.to_datetime(df[col] + pd.Timestamp("1900-01-01"), unit='D')
    #     return df
    

    @staticmethod
    def convert_back_dates(df, date_cols) -> pd.DataFrame:
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(date_cols, list), "date_cols must be a list"
        assert all(col in df.columns for col in date_cols), "All columns in date_cols must be in df.columns"

        for col in date_cols:
            # Ensure the column is numeric before adding the base date
            if not pd.api.types.is_numeric_dtype(df[col]):
                # raise TypeError(f"Column '{col}' must be numeric to perform date conversion.")
                print(f"WARNING: Column '{col}' must be numeric to perform date conversion.")
                # Convert to numeric if not already
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Handle NaN values by replacing them with a default date (e.g., 1970-01-01)
                # df[col].fillna(pd.Timestamp("1970-01-01").toordinal(), inplace=True)
                # Convert the numeric value back to datetime
                # df[col] = pd.to_datetime(df[col], unit='D', origin='unix')
            
            # Add the base date and convert back to datetime
            df[col] = pd.to_datetime(pd.to_timedelta(df[col], unit='D') + pd.Timestamp("1900-01-01"))
        return df
    

    # the function that applies identify_date_cols and convert_dates to the dataframe
    @staticmethod
    def convert_all_dates(data) -> Data:
        metadata = data.metadata
        # logger.info(f"Metadata of columns is \n {metadata.columns}")
        date_cols = Utils.identify_date_cols(data)
        # logger.info(f"Date columns are \n {date_cols}")
        df = Utils.convert_dates(data, date_cols)
        # logger.info(f"The head of preprocessed data frame is \n {df.head()}")
        data_obj = Data(df, metadata=metadata)
        return data_obj

    
    @staticmethod
    def impute(input_data) -> Data:
        # MICE = IterativeImputer(verbose=False)
        metadata = input_data.metadata
        imputer = IterativeImputer(
            # estimator='RandomForestRegressor',              # Default: BayesianRidge
            estimator=None,
            max_iter=10,                 # Maximum iterations per feature
            tol=0.001,                   # Stopping tolerance threshold
            random_state=42,             # Reproducibility                                                                                     
            initial_strategy='median',      # Initial imputation method
            keep_empty_features=True,          # Keep empty features TODO: check if this is needed
            skip_complete=True,              # Skip complete features
            )
        # set_trace() # breakpoint
        imp_data = imputer.fit_transform(input_data.data)
        # ensure that the imputed data is a DataFrame
        # debugging - breakpoint
        # import ipdb; ipdb.set_trace()

        if not isinstance(imp_data, pd.DataFrame) and isinstance(imp_data, np.ndarray):
            # Convert to DataFrame if not already
            imp_data = pd.DataFrame(imp_data, columns=input_data.data.columns)

        imp_data = Data(imp_data, metadata=metadata)
        return imp_data

    
    # its inverse function
    @staticmethod
    def convert_back_all_dates(data) -> Data:
        # FIX:
        # need to use metadata besides dataframe, or a data object
        # cannot identify columns with datetime w/o that
        metadata = data.metadata
        date_cols = Utils.identify_date_cols(data)
        df = Utils.convert_back_dates(data.data, date_cols)
        data_obj = Data(df, metadata=metadata)
        return data_obj


    # the method that identifies the categorical columns in the dataframe
    @staticmethod
    def identify_categorical_cols(data) -> list:
        # find all columns names that are categorical according to the metadata
        metadata = data.metadata
        categoric_cols = [col for col in data.data.columns if metadata.columns[col]['sdtype'] == 'categorical']
        return categoric_cols
    

    @staticmethod
    def identify_float_columns_that_are_booleans(df):
        float_cols = df.select_dtypes(include=["float"])  # Only float dtype columns
        boolean_like_cols = []
        for col in float_cols.columns:
            series_no_na = float_cols[col].dropna()
            # Check if all unique non-NA values are either 0.0 or 1.
            unique_vals = set(series_no_na.unique())
            if len(unique_vals) <= 2:
                boolean_like_cols.append(col)
        return boolean_like_cols


    @staticmethod
    def identify_float_columns_that_are_integers(df):                                                          
        float_cols = df.select_dtypes(include=["float"])  # Only float dtype columns                               
        integer_like_cols = []                                                                                     
        for col in float_cols.columns: # Drop missing values so we don't accidentally compare NaN                                  
            series_no_na = float_cols[col].dropna()                                                                
            # Check if all values are integers (fractional part is zero)                                           
            if (series_no_na % 1 == 0).all():                                                                      
                integer_like_cols.append(col)                                                                      
        return integer_like_cols
    

    @staticmethod
    def find_closest(value, discrete_vals) -> float:
        idx = np.argmin(np.abs(discrete_vals - value))
        return discrete_vals[idx]
    

    @staticmethod
    def generate_col_maps(data, categoric_cols) -> dict:
        df = data.data

        def make_labels_to_nums(df_col) -> dict:
            unique_labels = df_col.unique()
            unique_labels.sort()
            label2num = {i:label for i,label in enumerate(unique_labels)}
            return label2num
        
        col_maps = {}
        for col in categoric_cols:
            col_maps[col] = make_labels_to_nums(df[col])
        return col_maps
    

    @staticmethod
    def discretize_column(df, col_name, col_maps) -> pd.DataFrame:
        discrete_values = np.array(list(col_maps[col_name].keys()))
        df[col_name] = df[col_name].apply(lambda val: (val/max(df[col_name]) * max(discrete_values)))
        df[col_name] = df[col_name].apply(lambda val: Utils.find_closest(val, discrete_values))
        df[col_name] = df[col_name].apply(lambda val: col_maps[col_name][val])
        return df
    

    # apply the discretize_column function to all columns of interest in the dataframe df_forest
    @staticmethod
    def discretize_cols(df, categoric_cols, col_maps) -> pd.DataFrame:
        # df = data.data
        for col in categoric_cols:
            df_forest = Utils.discretize_column(df, col, col_maps)
        return df_forest
    

    @staticmethod
    def convert_types_pandas(data, metadata):
        # representations_map = {'Int':int, 'Float':float, 
        #                     'bool':bool, 'object':object}
        
        # sdtypes_map = {'numerical':pd.to_numeric,
        #                'categorical':'category',
        #                'datetime':pd.to_datetime}
        
        # Define conversion functions that operate on entire columns
        sdtypes_map = {
            'numerical': lambda col: pd.to_numeric(col),
            'categorical': lambda col: col.astype('category'),
            'boolean': lambda col: col.astype('boolean'),
            'id': lambda col: col.astype('category'),
            # 'datetime': lambda col: pd.to_datetime(col),
            'datetime': lambda col: col if pd.api.types.is_datetime64_any_dtype(col) else pd.to_datetime(col)
        }

        # Create a new DataFrame with converted types
        df_new_dict = {}
        for column in metadata.columns:
            # print(f"sdtype - {metadata.columns[column]['sdtype']}")
            try:
                df_new_dict[column] = sdtypes_map[metadata.columns[column]['sdtype']](data.data[column])
            except KeyError as e:
                print(f"KeyError: {e} for column {column}.")
                # Handle the case where the sdtype is not found
                # You can choose to skip or assign a default value
                # For now, we'll just skip it
                continue
        
        df_new = pd.DataFrame(df_new_dict)

        return df_new
    

    # the method to get categorical columns from the metadata
    @staticmethod
    def get_categorical_columns_meta(metadata) -> list:
        categoric_cols = [col for col in metadata.columns if metadata.columns[col]['sdtype'] == 'categorical']
        return categoric_cols
    
    # the method to get non_categorical columns from the metadata, id est - numerical
    @staticmethod
    def get_non_categorical_columns_meta(metadata) -> list:
        non_categoric_cols = [col for col in metadata.columns if metadata.columns[col]['sdtype'] != 'categorical']
        return non_categoric_cols
    

    # the function which analyzes the content of the given dataframe column and identifies if 
    # it is a categorical or numerical by checking unique values
    @staticmethod
    def identify_column_type(df, col_name) -> str:
        # Check if the column is numerical using pandas api
        if pd.api.types.is_numeric_dtype(df[col_name]):
            return 'numerical'
        # Check if the column is categorical using pandas api
        elif pd.api.types.is_categorical_dtype(df[col_name]):
            return 'categorical'
        # additional check if it is categorical by comparing unique values and all values in the column
        # if the values are string
        elif pd.api.types.is_string_dtype(df[col_name]):
            if df[col_name].nunique() < df[col_name].count():
                return 'categorical'
            else:
                return 'unknown'
        else:
            return 'unknown'

    # # the function that receives pandas dataframe and returns the list of columns that are categorical
    # @staticmethod
    # def get_categorical_columns(df) -> list:
    #     categoric_cols = [col for col in df.columns if df[col].dtype == 'category']
    #     return categoric_cols