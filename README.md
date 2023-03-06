# ASyH - Anonymous Synthesizer for Health Data.

## Overview

The ASyH is a software helping Clinics as holders of large quantities of highly restricted personal health data to provide the Medical Data Community with realistic datasets without the breach of privacy.  It does this by synthesizing data with Machine Learning techniques which preserve data distribution and correlation while adding as much variation to the synthetic data, in order for it to have no resemblance any of the original patient data entries.

For synthesis, metrics and quality assurance we will mainly use the [Synthetic Data Vault](https://sdv.dev) ([github](https://github.com/sdv-dev/SDV)).

## Usage

The most basic use case for ASyH is to create an ASyH Application object and call synthesize() to get a synthetic dataset from the best-performing SDV model (one of CopulaGAN, CTGAN, GaussianCopula, or TVAE [cf. [the SDV documentation](https://sdv.dev/SDV/api_reference/tabular/index.html)]).  The input original dataset should be provided as a pandas DataFrame, the synthesized dataset is output as pandas DataFrame as well.  For identification of numerical and categorical variables, a metadata file in JSON format needs to be provided (see below).

```python
import ASyH

asyh = ASyH.Application()
synthetic_data = asyh.synthesize('original_data.csv', metadata_file='metadata.json')

# write the synthetic dataset to CSV file:
with open('synthetic_data.csv', 'w', encoding='utf-8') as output_file:
    synthetic_data.to_csv(output_file)
```
Alternatively, you can specify an Excel file as first argument to `asyh.synthesize(.,.)`

Additionally, a report of the output data quality (in terms of similarity to the original data) can be generated with (appended to the above code, in the same script file)
```python
import pandas
import json

# We will need the original dataset as pandas DataFrame
original_data = pandas.read_csv('input_data.csv')

# We also need the metadata as a dict:
with open('metadata.json', 'r', encooding='utf-8') as md_file:
    metadata = json.load(md_file)

# the following will create the md file
#   report.md
# and, if an installation of TeXLive and pandoc is available
#   report.pdf
ASyH.report('report', asyh.model.model_type,
            original_data, synthetic_data,
            metadata['tables']['data'])
```

you will find a zip archive with all images, the markdow file (if generated the PDF as well), and the synthetic data in a CSV file.  Mind that the above code assumes that the metadata specifies the table name as 'data'.

## Metadata format

ASyH uses SDV's metadata format (cf. ['Metadata' in the SDV documentation](https://sdv.dev/SDV/developer_guides/sdv/metadata.html)).

The skeleton of the JSON file should look like the following
```JSON
{"tables":
    {"TABLE_NAME":
        {"fields":
            { ...field specifications...
            },
         "primary_key":...
        }
    }
}
```
Where `TABLE_NAME` should specify the table's name (this is only important for accessing the 'table metadata' in the corresponding python dict - metadata['tables']['TABLE_NAME']).  Specifying a `primary_key` is optional.

The `field specifications` are of the form

    "FIELD_NAME": {"type": "FIELD_TYPE"}
or

    "FIELD_NAME": {"type": "FIELD_TYPE", "subtype": "SUBTYPE"}

where `FIELD_NAME` is a field's (or data column's) name and `FIELD_TYPE` is on of `(numerical, datetime, categorical, boolean, id)`.  In the case of a `FIELD_TYPE` of `numerical`, the `SUBTYPE` should be specified as either `integer` or `float`.  For a `FIELD_TYPE` of `id` the `SUBTYPE` should be set to either `string` or `integer` according to the `FIELD_TYPE`'s format.

In case 'table metadata' is needed is required as argument (as to the `ASyH.report(...)` function), the data within the "tables"/"TABLE_NAME" hierarchy is to be used, i.e. when having read the metadata file into a dict `metadata`, use <nobr>`metadata['tables']['TABLE_NAME']`</nobr>, replacing `TABLE_NAME` with the actual name.

## Development

To do development on this software do this:

* Check out the repository
* Create a Python venv for the project
* Activate the venv
* Install the package editable (-e) with the test dependencies:

        pip install -e '. [tests]'

To run the tests set the PYTHONPATH and execute pytest on the 'tests' folder:

        export PYTHONPATH=$(pwd)
        pytest tests
