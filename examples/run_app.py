import json
import pandas

import ASyH


if __name__ == '__main__':
    # input/output file names
    FILENAME_ROOT = 'Kaggle_Sirio_Libanes-16features'

    INPUT_FILENAME = FILENAME_ROOT + '.xlsx'
    METADATA_FILE = FILENAME_ROOT + '.json'

    OUTPUT_NAME = FILENAME_ROOT + '-synth'

    OUTPUT_FILENAME = OUTPUT_NAME + '.csv'
    MODEL_FILENAME = OUTPUT_NAME + '.pkl'

    asyh = ASyH.Application()

    # Output 'synth' is a pandas dataframe
    synth = asyh.synthesize(INPUT_FILENAME)

    synth.to_csv(OUTPUT_FILENAME)

    # Save the model for later re-use to .pkl:
    asyh.model.save(MODEL_FILENAME)

    # Reporting
    # The following is needed for reporting
    real_data = pandas.read_excel(INPUT_FILENAME)
    with open(METADATA_FILE, 'r') as md_file:
        metadata = json.load(md_file)

    report = ASyH.Report(real_data, synth, metadata)
    report.generate(FILENAME_ROOT, asyh.model.model_type)
