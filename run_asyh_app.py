import json
import pandas as pd
import ASyH
import argparse
import logging
# import pdb

# creating a custom logger
logger = logging.getLogger("asyh_logger")
logger.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)

# Create a formatter and attach to handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Attach handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)


parser = argparse.ArgumentParser(prog="Run-asyh-app",
        description="Produces csv file with synthesized data, pickle file with saved synthesizer moderl, and json file with metadata",
        epilog='.')


parser.add_argument("--input_name_root", dest="INPUT_NAME_ROOT", type=str)
parser.add_argument("--input_format", dest="input_format", type=str)
parser.add_argument("--metadata_file", dest="METADATA_FILE", type=str)
parser.add_argument("--output_name_root", dest="OUTPUT_NAME_ROOT", type=str)


def readData(input_file, ext, **kwargs) -> pd.DataFrame:
    ext2method = {'csv':pd.read_csv, 'xlsx':pd.read_excel}
    return(ext2method[ext](input_file, **kwargs))


if __name__ == '__main__':
    # input/output file names
    args = parser.parse_args()
    print(f"Passed arguments are {args}")

    # pdb.set_trace()
    # FILENAME_ROOT = 'Kaggle_Sirio_Libanes-16features'
    INPUT_FILE = args.INPUT_NAME_ROOT + "." + args.input_format
    METADATA_FILE = args.METADATA_FILE
    # INPUT_FILENAME = INPUT_NAME_ROOT + '.xlsx'
    # METADATA_FILE = FILENAME_ROOT + '.json'

    OUTPUT_NAME = args.OUTPUT_NAME_ROOT + '-synth'
    OUTPUT_FILENAME = OUTPUT_NAME + '.csv'
    MODEL_FILENAME = OUTPUT_NAME + '.pkl'
    # pdb.set_trace()

    asyh = ASyH.Application()


    # Reporting
    # The following is needed for reporting
    # real_data = pandas.read_excel(INPUT_FILE)
    real_data = readData(INPUT_FILE, args.input_format, low_memory=False)
    logger.info("Data Frame has been loaded")   # Logs to console and file

    with open(METADATA_FILE, 'r') as md_file:
        metadata = json.load(md_file)

    logger.info("Metadata has been loaded from JSON file")   # Logs to console and file

    # Output 'synth' is a pandas dataframe
    synth = asyh.synthesize(INPUT_FILE, metadata=metadata, metadata_file=METADATA_FILE)
    logger.info("New data has been synthesized")   # Logs to console and file
    synth.to_csv(OUTPUT_FILENAME)
    logger.info("CSV file with synthesized data is saved")   # Logs to console and file
    # Save the model for later re-use to .pkl:
    asyh.model.save(MODEL_FILENAME)
    logger.info("The model checkpoint is saved.")   # Logs to console and file

    # report = ASyH.Report(real_data, synth, metadata)
    # report.generate(args.INPUT_NAME_ROOT, asyh.model.model_type)
