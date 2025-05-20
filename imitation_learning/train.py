import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt

import sys

sys.path.append(".")

import utils
from agent.bc_agent import BCAgent
from tensorboard_evaluation import Evaluation
import torch

train_losses = []
train_accuracies = []
valid_accuracies = []

def read_data(datasets_dir="./data", frac=0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, "data.pkl.gzip")

    f = gzip.open(data_file, "rb")
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = (
        X[: int((1 - frac) * n_samples)],
        y[: int((1 - frac) * n_samples)],
    )
    X_valid, y_valid = (
        X[int((1 - frac) * n_samples) :],
        y[int((1 - frac) * n_samples) :],
    )
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can train your model with discrete actions (as you get them from read_data) by discretizing the action space
    #    using action_to_id() from utils.py.

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96, 1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    print("... preprocessing data")

    X_train_gray = np.array([utils.rgb2gray(img).reshape(96, 96, 1) for img in X_train])
    X_valid_gray = np.array([utils.rgb2gray(img).reshape(96,96,1) for img in X_valid])
    y_train = np.array([utils.action_to_id(a) for a in y_train])
    y_valid = np.array([utils.action_to_id(a) for a in y_valid])

    if history_length>1:
        X_history = []
        y_history = []
        X_valid_history = []
        y_valid_history = []
        for idx in range(0, X_train_gray.shape[0], history_length):
            X_history.append(X_train_gray[idx:idx + history_length].reshape(96, 96, history_length))
            y_history.append(y_train[idx + history_length- 1])
        for idx in range(0, X_valid_gray.shape[0], history_length):
            X_valid_history.append(X_valid_gray[idx:idx + history_length].reshape(96, 96, history_length))
            y_valid_history.append(y_valid[idx + history_length - 1])
        return np.array(X_history), np.array(y_history), np.array(X_valid_history), np.array(y_valid_history)

    
    return X_train, y_train, X_valid, y_valid

def uniform_sampling(X_train,y_train):
    straight = len(y_train[y_train == 0])
    left = len(y_train[y_train == 1])
    right = len(y_train[y_train == 2])
    accelerate = len(y_train[y_train == 3])
    brake = len(y_train[y_train == 4])
    min_ = min(straight, left, right, accelerate)
    X_train_sampled = []
    y_train_sampled = []
    count_straight = 0
    count_left = 0
    count_right = 0
    count_accelerate = 0
    count_brake = 0
    for i in range(X_train.shape[0]):
        if (y_train[i] == 0 and count_straight <= 2*min_ ):
            X_train_sampled.append(X_train[i])
            y_train_sampled.append(y_train[i])
            count_straight += 1
        elif (y_train[i] == 1 and count_left <= min_):
            X_train_sampled.append(X_train[i])
            y_train_sampled.append(y_train[i])
            count_left += 1
        elif (y_train[i] == 2 and count_right <= min_):
            X_train_sampled.append(X_train[i])
            y_train_sampled.append(y_train[i])
            count_right += 1
        elif (y_train[i] == 3 and count_accelerate <= 4*min_):
            X_train_sampled.append(X_train[i])
            y_train_sampled.append(y_train[i])
            count_accelerate += 1
        elif (y_train[i] == 4 and count_brake <= min_):
            X_train_sampled.append(X_train[i])
            y_train_sampled.append(y_train[i])
            count_brake += 1
    return np.array(X_train_sampled), np.array(y_train_sampled)

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_minibatches,
    batch_size,
    lr,
    model_dir="./models",
    tensorboard_dir="./tensorboard",
    history_length=1,
):

    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    # TODO: specify your agent with the neural network in agents/bc_agent.py
    # agent = BCAgent(...)

    agent = BCAgent()
    agent.optimizer = torch.optim.Adam(agent.net.parameters(), lr=lr)

    tensorboard_eval = Evaluation(tensorboard_dir, "Imitation Learning", stats=["train_acc", "train_loss", "valid_acc"])

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training *during* the training in your web browser
    #
    # training loop
    # for i in range(n_minibatches):
    #     ...
    #     if i % 10 == 0:
    #         # compute training/ validation accuracy and write it to tensorboard
    #         tensorboard_eval.write_episode_data(...)


    def sample_minibatch(X, y, batch_size):
        indices = torch.randint(0, len(X), (batch_size,))
        return X[indices], y[indices]

    for i in range(n_minibatches):

        # Sample minibatch
        x_batch, y_batch = sample_minibatch(X_train, y_train, batch_size)

        train_loss = agent.update(x_batch, y_batch)

        if i % 10 == 0:
            y_pred = agent.predict(torch.tensor(x_batch).permute(0, 3, 1, 2))
            y_pred = y_pred.argmax(1)
            y_batch = torch.LongTensor(y_batch).view(-1)
            train_accuracy = (torch.sum(torch.eq(y_pred, y_batch)).item() / batch_size)
            print('train acc: ', train_accuracy)

            with torch.no_grad():
                y_valid_pred = agent.predict(torch.tensor(X_valid).permute(0, 3, 1, 2))
                y_valid_pred = y_valid_pred.argmax(1)
                y_valid = torch.LongTensor(y_valid).view(-1)
                valid_accuracy = torch.sum(torch.eq(y_valid_pred, y_valid)).item() / y_valid.shape[0]
                print('valid acc: ', valid_accuracy)

            eval_dict = {'train_acc': train_accuracy,
                         'train_loss': train_loss.item(),
                         'valid_acc': valid_accuracy, }
            tensorboard_eval.write_episode_data(i, eval_dict)

            print(f"[{i}/{n_minibatches}] Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
                  f"Valid Acc: {valid_accuracy:.4f}")

            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)
    
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, f"agent_{history_length}.pt"))
    print(f"Model saved in file: {model_dir}")


if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data")

    history_length = 5
    n_minibatches = 1000
    batch_size = 64
    lr = 1e-4

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(
        X_train, y_train, X_valid, y_valid, history_length=5
    )

    X_train, y_train = uniform_sampling(X_train, y_train)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid, n_minibatches=n_minibatches, batch_size=batch_size, lr=lr, history_length=history_length)


    #plot using matplotlib
    # Plot training loss
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, n_minibatches, 10), train_losses, label="Train Loss", color='blue')
    plt.title("Training Loss")
    plt.xlabel("Minibatch Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(0, n_minibatches, 10), train_accuracies, label="Train Accuracy", color='green')
    plt.plot(range(0, n_minibatches, 10), valid_accuracies, label="Valid Accuracy", color='red')
    plt.title("Accuracies")
    plt.xlabel("Minibatch Iteration")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join("./models", f"training_results_hist_{history_length}.png"))


