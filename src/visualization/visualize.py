import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../../data/interim/resampled_data.pkl')
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df['set'] == 1]
plt.plot(set_df['acc_y-axis'].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y-axis'].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()

for label in df['label'].unique():
    subset = df[df['label'] == label]
    fig, ax = plt.subplots()
    plt.plot(subset['acc_y-axis'][:100].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
label = 'squat'
participant = 'A'
category_df = df.query(f"label == '{label}'").query(
    f"participant == '{participant}'").reset_index(drop=True)

fig, ax = plt.subplots()
category_df.groupby(['category'])['acc_y-axis'].plot()
ax.set_ylabel('Acceleron y axis')
ax.set_xlabel("Samples")
plt.legend()
plt.title(f"Heavy vs Medium ({label}) for participant {participant}")
plt.show()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participant_df = df.query("label == 'bench'").sort_values(
    "participant").reset_index()
fig, ax = plt.subplots()
participant_df.groupby(['participant'])['acc_y-axis'].plot()
ax.set_ylabel('Acceleron y axis')
ax.set_xlabel("Samples")
plt.legend()
plt.title("Comparision b/w Participants")
plt.show()
# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = 'squat'
participant = 'A'
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[['acc_x-axis', "acc_y-axis", "acc_z-axis"]].plot(ax=ax)
ax.set_ylabel('Acceleron axis')
ax.set_xlabel("Samples")
plt.title(f"x,y,z axis of Acceleron ({label} for participant {participant})")

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[['acc_x-axis', "acc_y-axis", "acc_z-axis"]].plot(ax=ax)
            ax.set_ylabel('Acceleron axis')
            ax.set_xlabel("Samples")
            plt.title(
                f"x,y,z axis of Acceleron ({label} for participant {participant})")

for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[['gyr_x-axis', "gyr_y-axis", "gyr_z-axis"]].plot(ax=ax)
            ax.set_ylabel('Gyroscope axis')
            ax.set_xlabel("Samples")
            plt.title(
                f"x,y,z axis of Gyroscope ({label} for participant {participant})")


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------

label = 'row'
participant = 'A'
combined_plot_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)
fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
combined_plot_df[['acc_x-axis', "acc_y-axis", "acc_z-axis"]].plot(ax=ax[0])
combined_plot_df[['gyr_x-axis', "gyr_y-axis", "gyr_z-axis"]].plot(ax=ax[1])
plt.title("All combinations of Acceleron and Gyroscope")
ax[0].set_xlabel("Samples")
ax[1].set_xlabel("Samples")
ax[0].set_ylabel("Acceleron axis")
ax[1].set_ylabel("Gyroscope axis")
ax[0].legend()
ax[1].legend()
plt.show()


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------

labels = df['label'].unique()
participants = df['participant'].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
            combined_plot_df[['acc_x-axis', "acc_y-axis", "acc_z-axis"]].plot(ax=ax[0])
            combined_plot_df[['gyr_x-axis', "gyr_y-axis", "gyr_z-axis"]].plot(ax=ax[1])
            plt.title("All combinations of Acceleron and Gyroscope")
            ax[0].set_xlabel("Samples")
            ax[1].set_xlabel("Samples")
            ax[0].set_ylabel("Acceleron axis")
            ax[1].set_ylabel("Gyroscope axis")
            ax[0].legend()
            ax[1].legend()
            plt.savefig(f"../../reports/figures/{label}-participant-{participant}.png")
            plt.show()