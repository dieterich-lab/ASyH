# ASyH - Anonymous Synthesizer for Health Data (Release 1).

## Overview


A data protection tool that utilizes generative machine learning models to create synthetic datasets, safeguarding sensitive patient information.
The ASyH is a software helping Clinics as holders of large quantities of highly restricted personal health data to provide the Medical Data Community with realistic datasets without the breach of privacy.  It does this by synthesizing data with Machine Learning techniques which preserve data distribution and correlation while adding as much variation to the synthetic data, in order for it to have no resemblance to any of the original patient data entries.

For synthesis, metrics and quality assurance we will mainly use the [Synthetic Data Vault](https://sdv.dev) ([github](https://github.com/sdv-dev/SDV)).

## Installation and Upgrading

Using pip, the easiest way to install/upgrade ASyH is

    pip install --upgrade https://github.com/dieterich-lab/ASyH/tarball/v1.0.2

## Usage

The most basic use case for ASyH is to create an ASyH Application object and call synthesize() to get a synthetic dataset from the best-performing SDV model/synthesizer (one of CopulaGAN, CTGAN, GaussianCopula, or TVAE [cf. [the SDV documentation](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers)]).  The input original dataset should be provided as a pandas DataFrame, the synthesized dataset is output as pandas DataFrame as well.  For identification of numerical and categorical variables, a metadata file in JSON format needs to be provided (see below).


### With CMD interface of the launcher run_asyh_app.py
The following flags are available so far via the CMD of the launcher "run_asyh_app.py"
- "--input_name_root" : the name of the source table without its extension
- "--input_format" : what file format is used (CSV by default)
- "--metadata_file" : the name of the file with stored metadata
- "--output_name_root" : how the output file would be named
- "--to-preprocess" : boolean option, if the input shall be preprocessed, e.g. imputed missing values

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
