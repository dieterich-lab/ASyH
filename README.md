# ASyH - Anonymous Synthesizer for Health Data (Version 1.1.0).

## Overview

A data protection tool that utilizes generative machine learning models to create synthetic datasets, safeguarding sensitive patient information.
The ASyH is a software helping Clinics as holders of large quantities of highly restricted personal health data to provide the Medical Data Community with realistic datasets without the breach of privacy.  It does this by synthesizing data with Machine Learning techniques which preserve data distribution and correlation while adding as much variation to the synthetic data, in order for it to have no resemblance to any of the original patient data entries.

For synthesis, metrics and quality assurance we will mainly use the [Synthetic Data Vault](https://sdv.dev) ([github](https://github.com/sdv-dev/SDV)).

## Installation and Upgrading

Using pip, the easiest way to install/upgrade ASyH is

    pip install --upgrade https://github.com/dieterich-lab/ASyH/tarball/v1.1.0

## Usage

The most basic use case for ASyH is to create an ASyH Application object and call synthesize() to get a synthetic dataset from the best-performing SDV model/synthesizer (one of CopulaGAN, CTGAN, GaussianCopula, or TVAE [cf. [the SDV documentation](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)]).  The input original dataset should be provided as a pandas DataFrame, the synthesized dataset is output as pandas DataFrame as well.  For identification of numerical and categorical variables, a metadata file in JSON format needs to be provided (see below).


### CMD interface of the launcher launcher.py
The following flags are available so far via the CMD of the launcher "launcher.py"
- "--input_name_root" : the name of the source table without its extension
- "--input_format" : what file format is used (CSV by default)
- "--metadata_file" : the name of the file with stored metadata
- "--output_name_root" : how the output file would be named
- "--to-preprocess" : boolean option, if the input shall be preprocessed, e.g. imputed missing values
- "--longitudinal" : boolean option, if an app must produce also longitudinal information

#### Examples of command-line usages
The basic usage for making a new table based on reference tabular dataset:
```bash
$ python3 launcher.py --input_name_root src_table --input_format csv --metadata_file metadata_table.json --output_name_root new_table
```
If you need to make pre-processing pipeline work too:
```bash
$ python3 launcher.py --input_name_root src_table --input_format csv --metadata_file metadata_table.json --output_name_root new_table --preprocess
```

If you need to impose logical constraints on generated table, add the flag `--constraints`
```bash
$ python3 launcher.py --input_name_root src_table --input_format csv --metadata_file metadata_table.json --output_name_root new_table --constraints
```

### Usage with Python
```python
import ASyH

asyh = ASyH.Application()
synthetic_data = asyh.synthesize('original_data.csv', metadata_file='metadata.json')

# write the synthetic dataset to CSV file:
synthetic_data.to_csv(output_file, index=False)
```
Alternatively, you can specify an Excel file as first argument to `asyh.synthesize(.,.)`

Additionally, a report of the output data quality (in terms of similarity to the original data) can be generated with (appended to the above code, in the same script file)

```python
import ASyH
import pandas
import json

# We will need the original dataset as pandas DataFrame
original_data = pandas.read_csv('input_data.csv')

# We also need the metadata as a dict:
with open('metadata.json', 'r', encoding='utf-8') as md_file:
    metadata = json.load(md_file)

asyh = ASyH.Application()
synthetic_data = asyh.synthesize(input_data.csv', metadata_file='metadata.json')

# the following will create the md file
#   report.md
# and, if an installation of TeXLive and pandoc is available
#   report.pdf
report = ASyH.Report(original_data, synthetic_data, metadata)
report.generate('report', asyh.model.model_type)
```

you will find a zip archive with all images, the markdown file (if generated the PDF as well), and the synthetic data in a CSV file.  Mind that the above code assumes that the metadata specifies the table name as 'data'.

## Metadata format

ASyH uses SDV's metadata format (cf. ['Metadata' in the SDV documentation](https://docs.sdv.dev/sdv/reference/metadata-spec/single-table-metadata-json)).

The skeleton of the JSON file should look like the following
```JSON
{"columns":
    { ...column specifications...
    },
 "primary_key":...
}
```
Specifying a `primary_key` is optional.

The `column specifications` are of the form

    "COLUMN_NAME": {"sdtype": "COLUMN_TYPE"}

or

    "COLUMN_NAME": {"sdtype": "COLUMN_TYPE", "SPECIFIER": SPECIFIER_VALUE}

where `COLUMN_NAME` is a column variable's name and `COLUMN_TYPE` is on of `(numerical, datetime, categorical, boolean, id)`.  The `SPECIFIER`/`SPECIFIER_VALUE` pair to use depends on the `sdtype` of the variable, it does not apply to boolean and categorical variables, otherwise, they are:

* `computer_representation` for numerical variables.  
Allowed values are `"Float"`, `"Int8"`, `"Int16"`, `"Int32"`, `"Int64"`, `"UInt8"`, `"UInt16"`, `"UInt32"`, `"UInt64"`

* `regex_format` for `id` variables.  
The regex string should use Perl-style regular expression syntax (cf. also the [Python documentation](https://docs.python.org/3/library/re.html)).

* `datetime_format` is **required** for datetime type variables.  
The `SPECIFIER_VALUE` for this specifier is a string in [strftime format](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).

### The metadata format for longitudinal data
```JSON
{"columns":
    { ...column specifications...
    },
 "primary_key":...,
 "sequence_key":...
}
If you intend to produce longitudinal data, make sure that `sequence_key` is included in json file
```

If you are going to use autoregressive models, make sure metadata json file includes the following entries:

* `sequence_key` - a string label unique for a given time-series
* `relationships` - a list of dictionaries representing relations between parent and child tables, supposed to be used with multi-sequence models
* `column_relationships` - annotation of column groups based on higher level concepts, with that a new column category could be introduced, which might characterize several given columns. See the example below
```
{
    "type": "diet",
    "column_names": ["omnivore", "keto", "vegan", "vegetarian", "carnivore"]
}
```


## Pre-processing pipeline
Primary function is to fill in the NaNs with adequate values depending on the
column data type.  In case of an input table with lower entropy density, its
usage is more justifiable.  The core method is iterative imputation based on a
strategy for imputing missing values by modeling each feature with missing
values as a function of other features in a round-robin fashion.  The
implementation included in the Python module `fancyimpute` is used.


## Post-processing pipeline
To be implemented soon. Its main purpose is to detect and correct entries that do not
fit in the acceptable range of a column distribution, i.e. lies 1.5 IQR below
the first quartile or more than 1.5 IQR above the third quartile.  Additional
functionality is to carry out some meta-statistical analysis and to model
column distributions.


## Development

To do development on this software do this:

* Check out the repository
* Create a Python venv for the project
* Activate the venv
* Install the package editable (-e) with the test dependencies:

        pip install -e '.[tests]'

To run the tests set the PYTHONPATH and execute pytest on the 'tests' folder:

        export PYTHONPATH=$(pwd)
        pytest tests


## Release History
| Release | Date |
| ---: | ---: |
|1.0.0| 25/05/2023|
|1.1.0| 20/06/2025|


## Changelog - version 1.1.0

### Added

- **Forest Flow model** - for synthesizing tables based on diffusion architecture
- **Probabilistic Autoregressive model** - well suited for generating columns with time-series data
- The capability to synthesize data with provided logical constraints
- The support for longitudinal data generation (using PAR pipeline)
- As the part of pre-processing pipeline - the ability to impute missing values in a source table

### Changed

- Extended the capabilities of command-line interface, added more flags for generation constraints, longitudinal entries, and pre-processing pipeline
- Pre- and post-processing functional hooks