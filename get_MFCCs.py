#%%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import json


#%%
def get_mfcc(filepath):
    signal, sr = librosa.load(filepath, sr=None)
    # plt.figure(figsize=(14, 5))
    # librosa.display.waveplot(signal, sr=sr)
    # print(signal)
    mfcc = librosa.feature.mfcc(signal, sr=sr)
    # plt.plot(mfcc)
    return mfcc.tolist()


#%%
extracted_feature_mfcc = []

df = pd.read_csv('dataset.csv')
for row in df.itertuples():
    filepath = 'audio/' + row.filename
    mfcc = get_mfcc(filepath)
    extracted_feature_mfcc.append({
        "filename": row.filename,
        "category": row.category,
        "target": row.target,
        "mfcc": mfcc
    })

#%%

# print(extracted_feature_mfcc)

#%%
with open('dataset.json', 'w') as f:
    json.dump(extracted_feature_mfcc, f)

# %%
