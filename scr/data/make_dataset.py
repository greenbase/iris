# %%
from sklearn.datasets import load_iris
import pandas as pd
from utils import get_project_root

import os


data,target=load_iris(return_X_y=True,as_frame=True)
data_iris=pd.concat([data,target],axis=1)

root=get_project_root()


data_iris.to_csv(os.path.join(root,"data\\raw"))
