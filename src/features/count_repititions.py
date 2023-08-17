import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error
from DataTransformation import LowPassFilter

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../../data/interim/resampled_data.pkl')
df = df[df['label'] != 'rest']

r_acc = df['acc_x-axis']**2 + \
    df['acc_y-axis']**2 + df['acc_z-axis']**2
r_gyr = df['gyr_x-axis']**2 + \
    df['gyr_y-axis']**2 + df['gyr_z-axis']**2

df['acc_r'] = np.sqrt(r_acc)
df['gyr_r'] = np.sqrt(r_gyr)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df['label'] == 'bench']
deadlift_df = df[df['label'] == 'dead']
ohp_df = df[df['label'] == 'ohp']
squat_df = df[df['label'] == 'squat']
row_df = df[df['label'] == 'row']


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = bench_df

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_x-axis'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_y-axis'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_z-axis'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['acc_r'].plot()

plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_x-axis'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_y-axis'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_z-axis'].plot()
plot_df[plot_df['set'] == plot_df['set'].unique()[0]]['gyr_r'].plot()

# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000/200
LowPass = LowPassFilter()
# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df['set'] == bench_df['set'].unique()[0]]
deadlift_set = deadlift_df[deadlift_df['set'] == deadlift_df['set'].unique()[
    0]]
ohp_set = ohp_df[ohp_df['set'] == ohp_df['set'].unique()[0]]
squat_set = squat_df[squat_df['set'] == squat_df['set'].unique()[0]]
row_set = row_df[row_df['set'] == row_df['set'].unique()[0]]

column = 'acc_r'
LowPass.low_pass_filter(
    bench_set, column, fs, 0.4, order=10
)[column+'_lowpass'].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


def count_reps(dataset, cutoff=0.4, order=10, column='acc_r'):

    data = LowPass.low_pass_filter(
        dataset, column, fs, cutoff, order
    )
    indexes = argrelextrema(data[column+'_lowpass'].values, np.greater)
    peaks = data.iloc[indexes]

    fig, ax = plt.subplots()
    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")
    ax.set_ylabel(f"{column}_lowpass")
    exercise = dataset['label'].iloc[0].title()
    category = dataset['category'].iloc[0].title()
    plt.title(f"{category} {exercise}: {len(peaks)} Reps ")

    return len(peaks)


count_reps(bench_set)
# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df['reps'] = df['category'].apply(lambda x:5 if x == 'heavy' else 10)
rep_df = df.groupby(['label','category','set'])['reps'].max().reset_index()
rep_df['reps_pred'] = 0

for set in df['set'].unique():
    subset = df[df['set'] == set]

    column = 'acc_r'
    cutoff = 0.4

    if subset['label'].iloc[0] == 'squat':
        cutoff = 0.35
    
    if subset['label'].iloc[0] == 'row':
        cutoff = 0.65
        column = 'gyr_x-axis'

    if subset['label'].iloc[0] == 'ohp':
        cutoff = 0.35
    
    reps = count_reps(subset,cutoff=cutoff,column=column)
    rep_df.loc[rep_df['set'] == set,'reps_pred'] = reps


rep_df['reps_pred'] = rep_df['reps_pred'].astype(int)
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
error = mean_absolute_error(rep_df['reps'],rep_df['reps_pred']).round(2)

rep_df.groupby(['label','category'])[['reps','reps_pred']].mean().plot.bar()