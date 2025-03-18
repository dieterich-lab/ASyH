from ASyH.ctabgan_synthesizer import CTABGANSynthesizer
from ASyH.pipelines import CopulaGANPipeline, CTABGANPipeline
from ASyH.models import CTABGAN_Model
from ASyH.data import Data
from ASyH.metadata import Metadata
import json
import pandas as pd
import sdmetrics.reports
from ASyH.App import Application


# model = CTABGANSynthesizer()



# make the model get input data and run training and inference

def sdmetrics_quality(input_data, synth_data):
    report = sdmetrics.reports.single_table.QualityReport()
    report.generate(input_data.data,
                    synth_data.data,
                    input_data.metadata.metadata,
                    verbose=False)
    return report.get_score()


path_to_metadata = '../examples/Kaggle_Sirio_Libanes-16features.json'
with open(path_to_metadata, 'r') as fl:
    metadata = json.load(fl)

metadata = Metadata(metadata)

df = pd.read_excel('../examples/Kaggle_Sirio_Libanes-16features.xlsx')
data = Data(df)
data.set_metadata(metadata)
# import pdb; pdb.set_trace()

# pipe = CopulaGANPipeline(data)
pipe = CTABGANPipeline(data)


score_funcs = [pipe]
print(score_funcs)
# print(pipe.model)

app = Application()
print(app)
app._add_scoring(sdmetrics_quality, pipelines=score_funcs)
# app.synthesize()
pipe.run()