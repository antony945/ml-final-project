import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

def plot_avg_reward_from_csvs(filenames: dict, output_file: str = None):
    if not filenames:
        print("No CSV files provided.")
        return

    plt.figure(figsize=(12, 6))

    # Get color cycle
    # Use a Seaborn color palette (e.g., 'deep')
    color_palette = sns.color_palette('deep')
    # color_palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # color_palette = plt.cm.Set2.colors

    color_cycle = itertools.cycle(color_palette)
    used_colors = []

    # Read each CSV file and plot its reward and avg_reward
    for label, file in filenames.items():
        data = pd.read_csv(file)

        # Compute reward, avg_reward, best_reward per episode
        reward_per_episode = data.groupby('episode')['reward'].mean()
        avg_reward_per_episode = data.groupby('episode')['avg_reward'].mean()
        best_reward_per_episode = data.groupby('episode')['best_reward'].mean()
        
        # Get the color for avg_reward and use a lighter version for reward
        color = next(color_cycle)  # Get next color in cycle
        used_colors.append(color)

        # alternative_values = reward_per_episode
        alternative_values = best_reward_per_episode

        # Plot alternative values with lighter transparency
        plt.plot(alternative_values.index, alternative_values.values, color=color, alpha=1.0, linestyle=':')
        # Plot avg_reward with normal color
        plt.plot(avg_reward_per_episode.index, avg_reward_per_episode.values, color=color, label=f'{label}', linewidth=1.75)

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    # plt.title('Average Reward per Episode During TEST')
    # Put legend outside of the graph
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=6)
    # plt.legend(loc='best', framealpha=0.5)  # Semi-transparent background
    # plt.legend()
    # plt.grid(True)
    plt.grid(linestyle='--', alpha=0.7)

    # Use a logarithmic scale on the y-axis
    plt.yscale('log')

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()

    return used_colors

def plot_best_reward_from_csvs(filenames: dict, colors: list):
    if not filenames:
        print("No CSV files provided.")
        return

    plt.figure(figsize=(12, 6))

    # Read each CSV file and plot its reward and avg_reward
    for idx, (label, file) in enumerate(filenames.items()):
        data = pd.read_csv(file)

        # Compute best_reward per episode
        best_reward_per_episode = data.groupby('episode')['best_reward'].mean()
        
        # Get the color for avg_reward and use a lighter version for reward
        color = colors[idx]

        # Plot best_reward with normal color
        plt.plot(best_reward_per_episode.index, best_reward_per_episode.values, color=color, label=f'{label}')

    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Best Reward')
    plt.legend()
    plt.grid(True)

    # Use a logarithmic scale on the y-axis
    plt.yscale('log')

    plt.tight_layout()
    # plt.savefig("ciaotight.png")
    plt.show()

def plot_max_best_reward_bar(filenames: dict, colors: list, output_file: str = None):
    if not filenames:
        print("No CSV files provided.")
        return

    max_rewards = {}

    # Read each CSV file and find the max best reward
    for label, file in filenames.items():
        data = pd.read_csv(file)

        if 'reward' in data.columns:
            max_rewards[label] = data['reward'].max()
        else:
            print(f"Warning: 'reward' column not found in {file}")
            max_rewards[label] = None

    # Filter out None values (in case some files were missing the 'reward' column)
    max_rewards = {k: v for k, v in max_rewards.items() if v is not None}

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(max_rewards.keys(), max_rewards.values(), color=colors, alpha=1.0)

    # Labels and title
    plt.xlabel('Configurations')
    plt.ylabel('Max Best Reward')
    # plt.title('Maximum Best Reward for Each File')
    # plt.xticks(rotation=45)  # Rotate labels for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Use a logarithmic scale on the y-axis
    plt.yscale('log')

    # Adjust y-axis limits to provide more space at the top
    plt.ylim(bottom=plt.ylim()[0], top=max(max_rewards.values()) * 1.50)  # Increase upper limit by 40%

    # Show value on top of each bar
    for i, (label, value) in enumerate(max_rewards.items()):
        plt.text(i, value*1.05, f"{value:.2f}", ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if output_file is not None:
        plt.savefig(output_file)
    plt.show()

# Example usage:
filenames = {
    'Q-Basic': 'runs/FlappyBird_test_Q_LR=0.0001_DF=0.95_eDECAY=0.99995_eFIN=0.01_MEM=None_DIV=10_N=-1.csv',
    'Q-ER': 'runs/FlappyBird_test_Q_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_DIV=10_N=-1.csv',
    'DQN-Basic': 'runs/FlappyBird_test_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=False_duelDQN=False_HID=128_DEV=cuda_N=-1.csv',
    'DQN-Double': 'runs/FlappyBird_test_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=True_duelDQN=False_HID=128_DEV=cuda_N=-1.csv',
    'DQN-Dueling': 'runs/FlappyBird_test_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=False_duelDQN=True_HID=128_DEV=cuda_N=-1.csv',
    'DQN-DoubleDueling': 'runs/FlappyBird_test_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=True_duelDQN=True_HID=128_DEV=cuda_N=-1.csv',
}

colors = plot_avg_reward_from_csvs(filenames, "results/avg_reward_plot.png")
# plot_best_reward_from_csvs(filenames, colors)
plot_max_best_reward_bar(filenames, colors, "results/best_reward_bar_plot.png")