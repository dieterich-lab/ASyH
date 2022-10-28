from ASyH.data import Data
from ASyH.pipelines import CopulaGANPipeline, TVAEPipeline
from ASyH.dispatch import concurrent_dispatch

# source the real data
input_data = Data()
input_data.read('Kaggle_Sirio_Libanes_ICU_Prediction.xlsx')


# for now just use a dummy scoring method, which means, the return values
# displayed below will both be 2
def scoring(a, b):
    return 2

# creating two models
t = TVAEPipeline(input_data, scoring)
c = CopulaGANPipeline(input_data, scoring)

# run the models in the synthesizing and scoring pipelines
ret = concurrent_dispatch(t, c)

print(f"result: TVAE: {ret[0]},  CopulaGANPipeline: {ret[1]}")
