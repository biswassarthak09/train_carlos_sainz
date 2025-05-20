# export DISPLAY=:0 

import sys
sys.path.append(".") 

import numpy as np
import gym
from agent.dqn_agent import DQNAgent
from imitation_learning.agent.networks import CNN
from tensorboard_evaluation import *
from utils import EpisodeStats
import utils
from statistics import mean
import matplotlib.pyplot as plt


train_losses = []
train_rewards = []
straight_usage = []
left_usage = []
right_usage = []
accel_usage = []
brake_usage = []
episodes = []
eval_rewards_history = []

def run_episode(env, agent, deterministic,skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    loss = None
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    #print(state.shape)
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        # action_id = agent.act(...)
        # action = your_id_to_action_method(...)
        action_id = agent.act(state=state,env='CarRacing-v0',deterministic=deterministic)
        action = utils.id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            loss = agent.train(state, action_id, next_state, reward, terminal,env='CarRacing-v0')

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats, loss


def train_online(env, agent, num_episodes, history_length=0,max_timesteps = 1000, model_dir="./models_carracing", tensorboard_dir="./tensorboard"):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "Reinforcement Learning"),"train-car", ["episode_reward", "straight", "left", "right", "accel", "brake"])
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        skip = 0
        if i < int(num_episodes * 0.05):
            skip = 5
            max_timesteps = 250
        else:
            skip = 3
            max_timesteps = 1000
        
        stats, loss = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, do_training=True,skip_frames=skip)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                      "straight" : stats.get_action_usage(utils.STRAIGHT),
                                                      "left" : stats.get_action_usage(utils.LEFT),
                                                      "right" : stats.get_action_usage(utils.RIGHT),
                                                      "accel" : stats.get_action_usage(utils.ACCELERATE),
                                                      "brake" : stats.get_action_usage(utils.BRAKE)
                                                      })

        train_rewards.append(stats.episode_reward)
        straight_usage.append(stats.get_action_usage(utils.STRAIGHT))
        left_usage.append(stats.get_action_usage(utils.LEFT))
        right_usage.append(stats.get_action_usage(utils.RIGHT))
        accel_usage.append(stats.get_action_usage(utils.ACCELERATE))
        brake_usage.append(stats.get_action_usage(utils.BRAKE))
        train_losses.append(loss)
        episodes.append(i)


        # TODO: evaluate your agent every 'eval_cycle' episodes using run_episode(env, agent, deterministic=True, do_training=False) to 
        # check its performance with greedy actions only. You can also use tensorboard to plot the mean episode reward.
        # ...
        # if i % eval_cycle == 0:
        #    for j in range(num_eval_episodes):
        #       ...
        if i % eval_cycle == 0:
            eval_cycle_reward = []
            for j in range(num_eval_episodes):
                stats, eval_loss = run_episode(env,agent,deterministic=True,do_training=False)
                eval_cycle_reward.append(stats.episode_reward)
            print(mean(eval_cycle_reward))
            eval_rewards_history.append(mean(eval_cycle_reward))

        # store model.
        if i % eval_cycle == 0 or i >= (num_episodes - 1):
            agent.save(os.path.join(model_dir, "dqn_agent-.pt"))

    tensorboard.close_session()

def state_preprocessing(state):
    return utils.rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    num_eval_episodes = 5
    eval_cycle = 20
    num_episodes = 100
    history_length = 1000000

    env = gym.make('CarRacing-v0').unwrapped
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    Q = CNN()
    Q_target = CNN()
    agent = DQNAgent(Q, Q_target, num_actions=5, history_length=history_length)
    train_online(env, agent, num_episodes=num_episodes, history_length=0, model_dir="./models_carracing")


    
    plt.figure(figsize=(14, 10))

    # Plot Training Loss
    plt.subplot(2, 2, 1)
    if train_losses:
        plt.plot(episodes, train_losses, label="Loss", color='blue')
        plt.title("Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

    # Plot Training Rewards
    plt.subplot(2, 2, 2)
    if train_rewards:
        plt.plot(episodes, train_rewards, label="Episode Reward", color='green')
        plt.title("Training Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()

    # Plot Action Usage
    plt.subplot(2, 2, 3)
    if straight_usage:
        plt.plot(episodes, straight_usage, label="Straight Usage", color='red')
        plt.plot(episodes, left_usage, label="Left Usage", color='purple')
        plt.plot(episodes, right_usage, label="Right Usage", color='orange')
        plt.plot(episodes, accel_usage, label="Accel Usage", color='brown')
        plt.plot(episodes, brake_usage, label="Brake Usage", color='pink')
        plt.title("Action Usage (Straight, Left, Right, Accel, Brake)")
        plt.xlabel("Episode")
        plt.ylabel("Action Usage")
        plt.grid(True)
        plt.legend()

    # Plot Mean Evaluation Reward
    plt.subplot(2, 2, 4)
    if eval_rewards_history:
        plt.plot(range(0, len(eval_rewards_history) * eval_cycle, eval_cycle), eval_rewards_history, label="Mean Eval Reward", color='orange')
        plt.title("Mean Eval Reward")
        plt.xlabel("Episode")
        plt.ylabel("Mean Eval Reward")
        plt.grid(True)
        plt.legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.savefig("./models_carracing/training_results.png")
    # plt.show()