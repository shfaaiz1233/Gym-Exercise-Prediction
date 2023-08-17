import pandas as pd
from glob import glob


def read_data_from_files(files):
    # Accelerometer Data
    acc_df = pd.DataFrame()

    # Gyroscope Data
    gyr_df = pd.DataFrame()

    # Unique number for data from different files
    acc_set = 1
    gyr_set = 1

    for f in files:
        # Extracting Features
        participant = f.split('-')[0].replace(data_path, "")
        participant = participant[len(participant)-1]

        label = f.split('-')[1]

        category = f.split('-')[2].rstrip('123').rstrip('_MetaWear_2019')

        df = pd.read_csv(f)

        # Adding Features in DataFrame
        df['participant'] = participant
        df['label'] = label
        df['category'] = category

        # Adding data in Accelerometer and Gyroscope Data frames
        if 'Accelerometer' in f:
            df['set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])

        if 'Gyroscope' in f:
            df['set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])

    # Cleaning the data set
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

    del acc_df['epoch (ms)']
    del acc_df['elapsed (s)']
    del acc_df['time (01:00)']

    del gyr_df['epoch (ms)']
    del gyr_df['elapsed (s)']
    del gyr_df['time (01:00)']

    return acc_df, gyr_df


data_path = '../../data/raw/MetaMotion/'
files = glob(data_path + '*.csv')

acc_df, gyr_df = read_data_from_files(files)


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
data_merged.columns = [
    'acc_x-axis',
    'acc_y-axis',
    'acc_z-axis',
    'gyr_x-axis',
    'gyr_y-axis',
    'gyr_z-axis',
    'participant', 'label', 'category', 'set'
]
data_merged
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------
sampling = {
    'acc_x-axis': 'mean',
    'acc_y-axis': 'mean',
    'acc_z-axis': 'mean',
    'gyr_x-axis': 'mean',
    'gyr_y-axis': 'mean',
    'gyr_z-axis': 'mean',
    'participant': 'last',
    'label': 'last',
    'category': 'last',
    'set': 'last'
}

data_merged[:1000].resample(rule='200ms').apply(sampling)

#Splitting by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq='D'))]
resampled_data = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

resampled_data.info()

resampled_data['set'] = resampled_data['set'].astype('int')


# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

resampled_data.to_csv('../../data/interim/resampled_data.csv')
resampled_data.to_pickle('../../data/interim/resampled_data.pkl')