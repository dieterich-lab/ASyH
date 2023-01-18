# ASyH - Anonymous Synthesizer for Health Data.

## Overview

The ASyH is a software helping Clinics as holders of large quantities of highly restricted personal health data to provide the Medical Data Community with realistic datasets without the breach of privacy.  It does this by synthesizing data with Machine Learning techniques which preserve data distribution and correlation while adding as much variation to the synthetic data, in order for it to have no resemblance any of the original patient data entries.

For synthesis, metrics and quality assurance we will mainly use the [[https://sdv.dev][Synthetic Data Vault]] ([[https://github.com/sdv-dev/SDV][github]]).

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