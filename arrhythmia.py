# Load the dataset
from wfdb.io import record
import pandas as pd
import wfdb
record = record.rdrecord(r'C:\Users\sohan\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\109') # replace with your own ECG signal
#record =pd.read_csv(r'C:\Users\sohan\Downloads\samples.csv')
#annotations = pd.read_csv(r'C:\Users\sohan\Desktop\annotation.csv') # replace with your own annotations
ecgr=wfdb.rdann(r'C:\Users\sohan\Downloads\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\109',extension='atr')
annotations=ecgr.symbol
# Preprocess the signal

from biosppy.signals import ecg
import numpy as np

signal = record.p_signal[:,0]
fs = record.fs

# Filter the signal
filtered, _, _ = ecg.st.filter_signal(signal,sampling_rate =fs,order=4,frequency=30,ftype="FIR")

# Locate R-peaks
rpeaks = ecg.christov_segmenter(filtered, fs)
rpeakarray=np.array(rpeaks).astype("int32")
rpeakarray=rpeakarray.reshape(np.shape(rpeakarray)[1],)

# Extract heartbeats
heartbeats = []
for i in range(len(rpeakarray)-1):
    heartbeats.append(filtered[rpeakarray[i]:rpeakarray[i+1]])

# Extract features
features = []
for heartbeat in heartbeats:
    features.append([
        np.mean(heartbeat), # Mean of the heartbeat
        np.std(heartbeat), # Standard deviation of the heartbeat
        np.max(heartbeat), # Maximum value of the heartbeat
        np.min(heartbeat), # Minimum value of the heartbeat
        np.sqrt(np.mean(np.square(heartbeat))) # Root mean square of the heartbeat
])
# Train a machine learning model
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0)

annotations=np.array(annotations).reshape(-1,1)
svm.fit(features, annotations[0:len(features)])

# Predict whether the ECG signal is indicative of arrhythmia or not
test_heartbeat = heartbeats[0] # replace with your own test heartbeat
test_feature = [
    np.mean(test_heartbeat),
    np.std(test_heartbeat),
    np.max(test_heartbeat),
    np.min(test_heartbeat),
    np.sqrt(np.mean(np.square(test_heartbeat)))
]
prediction = svm.predict([test_feature])[0]

if prediction == 'N':
    print('No arrhythmia detected')
else:
    print('Arrhythmia detected')




