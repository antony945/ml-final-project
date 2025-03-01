import gymnasium as gym
from Agent import Agent, Q_Agent, DQN_Agent
import yaml
import flappy_bird_gymnasium
import argparse
import os
from datetime import datetime

# For every model, the name of the model is the key and the value is the name of the file in the models folder
MODELS_DICT = {
    "flappybird_q_none": "FlappyBird_training_Q_LR=0.0001_DF=0.95_eDECAY=0.99995_eFIN=0.01_MEM=None_DIV=10_N=1000000.pk1",
    "flappybird_q_er": "FlappyBird_training_Q_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_DIV=10_N=300000.pk1",
    "flappybird_dqn_er": "FlappyBird_training_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=False_duelDQN=False_HID=128_DEV=cuda_N=30000.pt",
    "flappybird_dqn_double_er": "FlappyBird_training_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=True_duelDQN=False_HID=128_DEV=cuda_N=30000.pt",
    "flappybird_dqn_dueling_er": "FlappyBird_training_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=False_duelDQN=True_HID=128_DEV=cuda_N=30000.pt",
    "flappybird_dqn_double_dueling_er": "FlappyBird_training_DQN_LR=0.0001_DF=0.95_eDECAY=0.9995_eFIN=0.01_MEM=ER_BATCH=128_dDQN=True_duelDQN=True_HID=128_DEV=cuda_N=30000.pt"
}

def test(hyperparameters: dict, n_episodes, show_render = True):
    # Initialise the environment
    env = gym.make(**hyperparameters['env_args'], render_mode=None if not show_render else "human")
    
    print(f"Action space: {env.action_space}")
    print(f"Obs space: {env.observation_space}")

    for episode in range(n_episodes):
        done = False
        obs, info = env.reset(seed=42)

        # play one episode
        i = 0
        while not done:
            # Render the current state of the game
            env.render()

            action = env.action_space.sample()
            # print(f"{episode} - {i}) Agent choose action: {action}")

            next_obs, reward, terminated, truncated, info = env.step(action)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

            i = i+1

    env.close()
    return

def run(hyperparameters: dict, n_episodes, dqn = False, is_training = False, show_plots = False, show_render = False, verbose = False, record_video = False, model_name: str = None, seed = None):
    # Initialise the environment
    env = gym.make(**hyperparameters['env_args'], render_mode="rgb_array" if not show_render else "human") # or "none"
    # env.metadata['render_fps'] = 120

    if is_training:
        training_period = 5000
        name_prefix = "training"
    else:
        training_period = 100
        name_prefix = "test"

    if record_video:
        strdate = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if model_name is not None:
            # erase extension from model name in presence of mulitple dots
            name_prefix = f"{model_name.rsplit(".", 1)[0]}_{name_prefix}"
        else:
            name_prefix = f"flappybird_{name_prefix}"

        env = gym.wrappers.RecordVideo(env, video_folder="videos", name_prefix=name_prefix, episode_trigger=lambda x: x % training_period == 1)
    # env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=hyperparameters['n_episodes'])

    if dqn:
        agent = DQN_Agent(
            env,
            hyperparameters
        )
    else:
        agent = Q_Agent(
            env,
            hyperparameters,
        )
    
    agent.run(n_episodes, is_training, show_plots, verbose, model_name, seed=seed)
    return agent.model_name

def main(hyperparameter_set: str, test_model: bool = False):
    # Load hyperparameters set
    with open('hyperparameters.yml', 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameter_set]

    # TEST
    # test(hyperparameters, 3, show_render=True)

    seed = hyperparameters.get('seed', None)
    DQN = hyperparameters.get('dqn', True)

    if test_model:
        model_name = MODELS_DICT[hyperparameter_set]
        # RENDER
        run(hyperparameters, 1, dqn=DQN, is_training=False, show_plots=False, show_render=False, verbose=False, record_video=False, model_name=model_name, seed=seed)
        return

    # TRAINING
    model_name = run(hyperparameters, hyperparameters.get('n_episodes', None), dqn=DQN, is_training=True, show_plots=True, show_render=False, verbose=True, record_video=False, model_name=None, seed=seed)

    model_name = f"models/{model_name}"
    # RUN
    run(hyperparameters, 1000, dqn=DQN, is_training=False, show_plots=True, show_render=False, verbose=False, record_video=False, model_name=model_name, seed=seed)

    # RENDER
    run(hyperparameters, 5, dqn=DQN, is_training=False, show_plots=False, show_render=True, verbose=False, record_video=False, model_name=model_name, seed=seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the RL agent with specified parameters.')
    parser.add_argument('--q', action='store_true', help='Use Q-learning agent')
    parser.add_argument('--dqn', action='store_true', help='Use DQN agent')
    parser.add_argument('--er', action='store_true', help='Use Experience Replay')
    # parser.add_argument('--per', action='store_true', help='Use Prioritized Experience Replay') # NOT FULLY TESTED
    parser.add_argument('--double', action='store_true', help='Use DoubleDQN')
    parser.add_argument('--dueling', action='store_true', help='Use DuelingDQN')
    parser.add_argument('--test', action='store_true', help='Load and test with rendering a previously trained model')

    args = parser.parse_args()

    if args.q:
        agent_type = 'q'
    elif args.dqn:
        agent_type = 'dqn'
        if args.double:
            agent_type += '_double'
        if args.dueling:
            agent_type += '_dueling'
    else:
        raise ValueError("You must specify either --q or --dqn")

    if args.er:
        replay_type = 'er'
    # elif args.per:
        # replay_type = 'per'
    else:
        replay_type = 'none'

    hyperparameter_set = f"flappybird_{agent_type}_{replay_type}"
    
    main(hyperparameter_set, test_model=args.test)