'Example ASyH script to demonstrate building a customized pipeline with\
 ASyH\'s low-level API'

from ASyH import Data, Metadata, concurrent_dispatch
from ASyH.pipelines import CopulaGANPipeline, TVAEPipeline
from ASyH.metrics.bivariate_statistics import pc_comparison


# source the real data
input_data = Data()
input_data.read('Kaggle_Sirio_Libanes-16features.xlsx')

metadata = Metadata()
metadata.read('Kaggle_Sirio_Libanes-16features.json')

input_data.set_metadata(metadata)

# creating two models
t = TVAEPipeline(input_data)
t.add_scoring(pc_comparison)
c = CopulaGANPipeline(input_data)
c.add_scoring(pc_comparison)

# run the models in the synthesizing and scoring pipelines
ret = concurrent_dispatch(t, c)

print(f"result: TVAE: {ret[0]},  CopulaGANPipeline: {ret[1]}")
