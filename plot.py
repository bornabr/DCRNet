import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

# Read the CSV file
csv_path = 'results.csv'
df = pd.read_csv(csv_path)

print(df.head())

def plot_scenario(data, title, xlabel, ylabel, ax, scenario):
    ax.plot(data[:, 0], data[:, 1], label=scenario)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    

# Convert the 'scenario' column to a categorical column
df['scenario'] = pd.Categorical(df['scenario'])

# Iterate through the rows of the DataFrame, grouped by scenario
for scenario, scenario_group in df.groupby('scenario'):
    # Create a folder for each scenario
    scenario_folder = f'./plots/{scenario}'
    os.makedirs(scenario_folder, exist_ok=True)

    # Create separate figures for each scalar
    fig_train_loss, ax_train_loss = plt.subplots()
    fig_train_lr, ax_train_lr = plt.subplots()
    fig_val_loss, ax_val_loss = plt.subplots()
    fig_val_nmse, ax_val_nmse = plt.subplots()

    for _, row in scenario_group.iterrows():
        version = row['version']
        expansion = row['expansion']
        reduction = row['reduction']
        log_dir = f'./runs/{version}'

        # Load the TensorBoard log
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Get the scalar data
        train_loss = event_acc.Scalars('Train/Loss')
        train_lr = event_acc.Scalars('Train/LR')
        val_loss = event_acc.Scalars('Validation/Loss')
        val_nmse = event_acc.Scalars('Validation/NMSE')

        # Convert scalar data to arrays
        train_loss_data = np.array([[e.step, e.value] for e in train_loss])
        train_lr_data = np.array([[e.step, e.value] for e in train_lr])
        val_loss_data = np.array([[e.step, e.value] for e in val_loss])
        val_nmse_data = np.array([[e.step, e.value] for e in val_nmse])

        # Create a label for each version using expansion and reduction values
        label = fr'$\times {expansion}, \eta = {reduction}$'

        # Plot each version in separate figures
        plot_scenario(train_loss_data, 'Train Loss', 'Step', 'Loss', ax_train_loss, label)
        plot_scenario(train_lr_data, 'Train Learning Rate', 'Step', 'Learning Rate', ax_train_lr, label)
        plot_scenario(val_loss_data, 'Validation Loss', 'Step', 'Loss', ax_val_loss, label)
        plot_scenario(val_nmse_data, 'NMSE', 'Step', 'NMSE', ax_val_nmse, label)

    # Save the separate plots
    plt.figure(fig_train_loss.number)
    plt.tight_layout()
    plt.savefig(f'{scenario_folder}/train_loss.png')
    plt.close()

    plt.figure(fig_train_lr.number)
    plt.tight_layout()
    plt.savefig(f'{scenario_folder}/train_lr.png')
    plt.close()

    plt.figure(fig_val_loss.number)
    plt.tight_layout()
    plt.savefig(f'{scenario_folder}/val_loss.png')
    plt.close()

    plt.figure(fig_val_nmse.number)
    plt.tight_layout()
    plt.savefig(f'{scenario_folder}/val_nmse.png')
    plt.close()
