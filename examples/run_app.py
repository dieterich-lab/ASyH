import pathlib

from ASyH.App import Application


if __name__ == '__main__':
    asyh = Application()

    FILENAME_ROOT = 'Kaggle_Sirio_Libanes-16features'

    INPUT_FILENAME = FILENAME_ROOT + '.xlsx'

    OUTPUT_NAME = FILENAME_ROOT + '-synth'

    OUTPUT_FILENAME = OUTPUT_NAME + '.csv'
    MODEL_FILENAME = OUTPUT_NAME + '.pkl'

    synth = asyh.synthesize(INPUT_FILENAME)

    # synth is a pandas dataframe
    synth.to_csv(OUTPUT_FILENAME)
    asyh.model.save(MODEL_FILENAME)
