# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:11:28 2023

@author: Mauro Namías - Fundación Centro Diagnóstico Nuclear - mnamias@fcdn.org.ar
SNMMI AI Taskforce - FDG PET/CT Radiomics Machine Learning Challenge 2023
"""
# %% 
import numpy as np
import pandas as pd
from joblib import load
from sklearn.impute import KNNImputer
import torch
import torch.nn as nn


#%% Inference on blind test dataset

# Load trained models
 ## Load trained model for inference
checkpoint = torch.load('./AE_10f_v2.dat')
torch.cuda.empty_cache()


class Autoencoder(nn.Module):
    """Makes the main denoising auto

    Parameters
    ----------
    in_shape [int] : input shape
    enc_shape [int] : desired encoded shape
    """

    def __init__(self, in_shape, enc_shape):
        super(Autoencoder, self).__init__()
        
        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, enc_shape),
        )
        
        self.decode = nn.Sequential(
            nn.BatchNorm1d(enc_shape),
            nn.Linear(enc_shape, 64),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, in_shape)
        )
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x


device = ('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Autoencoder(in_shape=504, enc_shape=10).double().to(device)


encoder.load_state_dict(checkpoint['model_state_dict'])
del checkpoint
encoder.eval()


scaler = load('scaler.joblib')
regr = load('AE10f_RF_v2.joblib')
imputer = KNNImputer(n_neighbors=5)

# %%
# Load Training dataset
data = pd.read_excel('./dataset.xlsx')

# Select only data with events for PFS regression
data["Event"] = data["Event"].astype(bool)
data_events = data[data["Event"]==True]
X_train = data_events.iloc[:,3:507]

data = pd.read_excel('./SNMMI_CHALLENGE_TESTING_V01112023.xlsx')
X_test = data.iloc[:,1:]
X_test.rename(columns = {'ExactVolume':'ExactVolume (uL)'}, inplace = True) 

frames = [X_train, X_test]
frames = pd.concat(frames)

columns_with_nan = frames.columns[frames.isna().any()].tolist()

for covariate in columns_with_nan:
    print(covariate)  
    frames[covariate] = imputer.fit_transform(frames[covariate].values.reshape(-1, 1))

X_test = frames.iloc[90:,]

X_test = scaler.transform(X_test)

x = torch.from_numpy(X_test).to(device)


with torch.no_grad():
    encoded = encoder.encode(x)

enc = encoded.cpu().detach().numpy()

y_pred = regr.predict(enc)

# dump results to csv
np.savetxt("AE10f_RF_v2_results.csv", y_pred, delimiter=",")
