#%%
import json
import math

import librosa.display
import pandas as pd

#%%
SAMPLE_RATE = 22050
TRACK_DURATION = 5 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
num_segments = 5
samples_per_segment = SAMPLES_PER_TRACK / num_segments
hop_length = 512
num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

#%%
extracted_feature_mfcc = []

df = pd.read_csv('dataset_all.csv')

for row in df.itertuples():
    file_path = 'audio/' + row.filename

    signal, sr = librosa.load(file_path, sr=None)

    for i in range(5):
        seg = i + 1
        start = int(samples_per_segment) * seg
        finish = start + int(samples_per_segment)
        extracted_feature_mfcc.append({
            "filename": row.filename,
            "category": row.category,
            "target": row.target,
            "mfcc": librosa.feature.mfcc(signal[start : finish], sr=sr / 2).tolist()
        })



#%%

# print(extracted_feature_mfcc)

#%%
with open('dataset_all.json', 'w') as f:
    json.dump(extracted_feature_mfcc, f)

# %%
