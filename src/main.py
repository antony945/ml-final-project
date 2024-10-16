import gymnasium as gym

def run():
    # Initialise the environment
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human")

    # Reset the environment to generate the first observation
    state = env.reset(seed=42)[0] # states: 0 to 63, 0=top left corner, 63=bottom right corner
    terminated = False # True when goal is reached
    truncated = False # True when it reach max actions available, set a limit on no. of actions

    while not (terminated or truncated):
        # this is where you would insert your policy
        action = env.action_space.sample() # 0=left, 1=down, 2=right, 3=up

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        new_state, reward, terminated, truncated, info = env.step(action)

        state = new_state

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == '__main__':
    run()