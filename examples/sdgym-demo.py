#!/usr/bin/env python3
"""Run ASyH in SDGym"""

import sdgym

from ASyH.App import Application
from ASyH.metadata import Metadata
from ASyH.data import Data


def asyh_synthesizer(real_data, metadata):
    """SDGym adaptation of ASyH."""
    asyh_app = Application()
    table_name = metadata.get_tables()[0]
    asyh_metadata = Metadata(metadata=metadata.to_dict())
    input_data = Data(real_data[table_name])
    asyh_app.train(input_data, metadata=asyh_metadata)
    num_rows = len(real_data[table_name])
    return (table_name, num_rows, asyh_app)


def sample_asyh(synthesizer, num_samples):
    table_name, num_rows, generator = synthesizer
    return {table_name:
            generator.synthesize(num_rows)}


scores = sdgym.benchmark_single_table(
    synthesizers=(asyh_synthesizer, sample_asyh),
    sdv_datasets=['alarm'])

scores.to_csv('sdgym-asia-asyh-scores.csv')
