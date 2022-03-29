import pandas as pd
import numpy as np
from braingeneers import datasets_electrophysiology as de

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sbn

# take data coming out from load_data and transform it
X, fs, num_frames = de.load_data(batch_uuid='2021-10-05-e-org1_real', experiment_num=0, channels=[0, 1, 2], offset=20,
                                 length=15000)
X = StandardScaler().fit_transform(X)

col_names = ['x' + str(col) for col in range(0, X.shape[1])]
df = pd.DataFrame(X, columns=col_names)
if df.empty:
    print('empty')
print(df.head())
