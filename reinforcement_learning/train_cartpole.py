import sys

sys.path.append(".")

import numpy as np
import gym
import itertools as it
from agent.dqn_agent import DQNAgent
from tensorboard_evaluation import *
from agent.networks import MLP
from utils import EpisodeStats
import matplotlib.pyplot as plt


train_losses = []
train_rewards = []
a_0_usage = []
a_1_usage = []
eval_rewards_history = []
episodes = []


def run_episode(
    env, agent, deterministic, do_training=True, rendering=False, max_timesteps=1000
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()  # save statistics like episode reward or action usage
    state = env.reset()

    step = 0
    loss = None
    while True:

        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        if do_training:
            loss = agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal or step > max_timesteps:
            break

        step += 1

    return stats, loss


def train_online(
    env,
    agent,
    num_episodes,
    model_dir="./models_cartpole",
    tensorboard_dir="./tensorboard",
):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train agent")

    tensorboard = Evaluation(
        os.path.join(tensorboard_dir, "Reinforcement Learning"), 'rl', ["episode_reward", "a_0", "a_1", "mean_eval_reward"]
    )

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats, loss = run_episode(env, agent, deterministic=False, do_training=True)
        tensorboard.write_episode_data(
            i,
            eval_dict={
                "episode_reward": stats.episode_reward,
                "a_0": stats.get_action_usage(0),
                "a_1": stats.get_action_usage(1),
            },
        )

        train_rewards.append(stats.episode_reward)
        a_0_usage.append(stats.get_action_usage(0))
        a_1_usage.append(stats.get_action_usage(1))
        train_losses.append(loss)
        episodes.append(i)

        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...

        if i % eval_cycle == 0:
            eval_rewards = []
            for j in range(num_eval_episodes):
                eval_stats, eval_loss = run_episode(env, agent, deterministic=True, do_training=False)
                eval_rewards.append(eval_stats.episode_reward)
            mean_eval_reward = np.mean(eval_rewards)
            tensorboard.write_episode_data(
                i,
                eval_dict={
                    "mean_eval_reward": mean_eval_reward,
                },
            )
            eval_rewards_history.append(mean_eval_reward)

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent_1000000.pt"))

    tensorboard.close_session()


if __name__ == "__main__":

    num_eval_episodes = 5  # evaluate on 5 episodes
    eval_cycle = 20  # evaluate every 10 episodes

    # You find information about cartpole in
    # https://github.com/openai/gym/wiki/CartPole-v0
    # Hint: CartPole is considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

    env = gym.make("CartPole-v0").unwrapped

    state_dim = 4
    num_actions = 2
    history_length = 1000000
    num_episodes = 1000

    # TODO:
    # 1. init Q network and target network (see dqn/networks.py)
    # 2. init DQNAgent (see dqn/dqn_agent.py)
    # 3. train DQN agent with train_online(...)

    q_network = MLP(state_dim, num_actions)
    target_network = MLP(state_dim, num_actions)
    agent = DQNAgent(q_network, target_network, num_actions=num_actions, history_length=history_length)
    train_online(env, agent, num_episodes=num_episodes)



# Plot training loss
    plt.figure(figsize=(10, 6))

    # Plot Loss
    plt.subplot(2, 2, 1)
    if train_losses:
        plt.plot(episodes, train_losses, label="Loss", color='blue')
        plt.title("Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

    # Plot training rewards
    plt.subplot(2, 2, 2)
    plt.plot(episodes, train_rewards, label="Episode Reward", color='green')
    plt.title("Training Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()

    # Plot action usage (a_0 and a_1)
    plt.subplot(2, 2, 3)
    plt.plot(episodes, a_0_usage, label="Action 0 Usage", color='red')
    plt.plot(episodes, a_1_usage, label="Action 1 Usage", color='purple')
    plt.title("Action Usage (a_0, a_1)")
    plt.xlabel("Episode")
    plt.ylabel("Action Usage")
    plt.grid(True)
    plt.legend()

    # Plot mean evaluation reward
    plt.subplot(2, 2, 4)
    plt.plot(range(0, num_episodes, eval_cycle), eval_rewards_history, label="Mean Eval Reward", color='orange')
    plt.title("Mean Eval Reward")
    plt.xlabel("Episode")
    plt.ylabel("Mean Eval Reward")
    plt.grid(True)
    plt.legend()

    # Adjust layout and show plots
    plt.tight_layout()
    # plt.show()
    plt.savefig("./models_cartpole/training_results.png")
