import sdv.metadata as metadata
from ASyH.metadata import Metadata
import json
import pandas as pd


INPUT_FILE_CSV = "/home/gsergei/ASyH/tiny_ukbb_subset_0to1000rows.csv"
OUTPUT_FILE_JSON = "/home/gsergei/ASyH/metadata_tiny_ukbb_0to1k.json"

meta_sdv = metadata.SingleTableMetadata()
meta_sdv.detect_from_csv(INPUT_FILE_CSV)
meta_dict = meta_sdv.dict() # where metadata is stored in dictionary

with open(OUTPUT_FILE_JSON, "w") as meta_file:
    json.dump(meta_dict, meta_file)
