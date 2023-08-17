import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle('../../data/interim/outliers_removed.pkl')
predictor_columns = list(df.columns[:6])

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2
# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df['set'] == 25]['acc_y-axis'].plot()
df[df['set'] == 50]['acc_y-axis'].plot()

duration = df[df['set'] == 1].index[-1] - df[df['set'] == 1].index[0]
duration.seconds

for s in df['set'].unique():
    start = df[df['set'] == s].index[0]
    stop = df[df['set'] == s].index[-1]

    duration = stop - start

    df.loc[df['set'] == s, "duration"] = duration.seconds


duration_df = df.groupby(['category'])['duration'].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
LowPass = LowPassFilter()
fs = 1000 / 200
cutoff = 1.3
df_lowpass = LowPass.low_pass_filter(
    df_lowpass, 'acc_y-axis', fs, cutoff, order=5)


subset = df_lowpass[df_lowpass['set'] == 45]
print(subset['label'][0])

fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
ax[0].plot(subset['acc_y-axis'].reset_index(drop=True), label="raw data")
ax[1].plot(subset['acc_y-axis_lowpass'].reset_index(drop=True),
           label="Butterworth filter")
ax[0].legend()
ax[1].legend()


for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col+'_lowpass']
    del df_lowpass[col+'_lowpass']


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
plt.figure(figsize=(20, 10))
plt.plot(range(1, len(predictor_columns)+1), pc_values)
plt.xlabel("Principal component number")
plt.ylabel("Explained Variance")
plt.show()

df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

subset = df_pca[df_pca['set'] == 35]

subset[['pca_1', 'pca_2', 'pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

r_acc = df_squared['acc_x-axis']**2 + \
    df_squared['acc_y-axis']**2 + df_squared['acc_z-axis']**2
r_gyr = df_squared['gyr_x-axis']**2 + \
    df_squared['gyr_y-axis']**2 + df_squared['gyr_z-axis']**2

df_squared['acc_r'] = np.sqrt(r_acc)
df_squared['gyr_r'] = np.sqrt(r_gyr)

subset = df_squared[df_squared['set'] == 25]


subset[['acc_r', 'gyr_r']].plot(subplots=True)

# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
df_temporal = df_squared.copy()
predictor_columns += ['acc_r', 'gyr_r']

NumAbs = NumericalAbstraction()
ws = int(1000/200)

for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'mean')
    df_temporal = NumAbs.abstract_numerical(df_temporal, [col], ws, 'std')

df_temporal_list = []
for set in df_temporal['set'].unique():
    subset = df_temporal[df_temporal['set'] == set].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(subset, [col], ws, 'mean')
        subset = NumAbs.abstract_numerical(subset, [col], ws, 'std')
    df_temporal_list += [subset]

df_temporal = pd.concat(df_temporal_list)

subset[['acc_y-axis', 'acc_y-axis_temp_mean_ws_5',
        'acc_y-axis_temp_std_ws_5']].plot()
plt.title("Accelerometer")
plt.show()
subset[['gyr_y-axis', 'gyr_y-axis_temp_mean_ws_5',
        'gyr_y-axis_temp_std_ws_5']].plot()
plt.title("Gyroscope")
plt.show()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
freqAbs = FourierTransformation()

fs = int(1000/200)
ws = int(2800/200)

df_freq = freqAbs.abstract_frequency(df_freq, ['acc_y-axis'], ws, fs)

len(df_freq.columns)

subset = df_freq[df_freq['set'] == 15]
subset[['acc_y-axis']].plot()
subset[
    [
        'acc_y-axis_max_freq',
        'acc_y-axis_freq_weighted',
        'acc_y-axis_pse',
        'acc_y-axis_freq_1.429_Hz_ws_14',
        'acc_y-axis_freq_2.5_Hz_ws_14'
    ]
].plot()

df_freq_list = []

for set in df_freq['set'].unique():
    print(f"Applying Fourier Transformation to set {set}")
    subset = df_freq[df_freq['set'] == set].reset_index(drop=True).copy()
    subset = freqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list += [subset]

df_freq = pd.concat(df_freq_list).set_index('epoch (ms)', drop=True)

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()
cluster_columns = ['acc_x-axis', 'acc_y-axis', 'acc_z-axis']
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias += [kmeans.inertia_]

plt.figure(figsize=(10,10))
plt.plot(k_values,inertias)
plt.xlabel("K")
plt.ylabel("Sum of squared distances")
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster['cluster'] = kmeans.fit_predict(subset)

#Ploting clusters

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for c in df_cluster['cluster'].unique():
    subset = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset['acc_x-axis'],subset['acc_y-axis'],subset['acc_z-axis'],label=c)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.legend()
plt.title("On the basis of clusters")
plt.show()


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
for l in df_cluster['label'].unique():
    subset = df_cluster[df_cluster['label'] == l]
    ax.scatter(subset['acc_x-axis'],subset['acc_y-axis'],subset['acc_z-axis'],label=l)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.legend()
plt.title("On the basis of labels")
plt.show()


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

df_cluster.to_pickle('../../data/interim/data_features.pkl')
df_cluster.to_csv('../../data/interim/data_features.csv')
