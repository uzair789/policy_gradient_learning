import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

import gym

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

OUTPUT_PATH = './results'
ACTION_DIM = 4
STATE_DIM = 8


def create_model():
    """ This function create the NN using keras library.
    """

    model = Sequential()
    model.add(Dense(16,
                    input_shape=(STATE_DIM,),
                    activation='relu',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    model.add(Dense(ACTION_DIM,
                    activation='softmax',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    return model

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr, gamma, env):
        self.env = env
        self.model = model
        self.gamma = gamma
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr),
                           metrics=['accuracy'])


    def get_G(self, rewards):
        G = np.zeros(len(rewards))
        temp = 0
        for i in range(len(G)-1, -1, -1):
            G[i] = rewards[i]*0.01 + self.gamma*temp
            temp = G[i]

        # Normalize G
        G = (G - np.mean(G)) / np.std(G)
        return G


    def get_train_batch(self, states, actions, rewards):
        assert len(states) == len(actions)
        action_batch = np.zeros([len(actions), ACTION_DIM])
        action_batch[np.arange(len(actions)), actions] = 1

        G = self.get_G(rewards)
        G_batch = np.zeros([len(G), ACTION_DIM])
        G_batch[np.arange(len(G_batch)), actions] = G

        return G_batch, action_batch
        
  

    def sample_action(self, action_prob):
        # Creating greedy policy for test time.
        actions = [x for x in range(ACTION_DIM)] 
        action = np.random.choice(actions ,p=np.squeeze(action_prob))
        return action
	
	




    def train(self, env):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env, train=True)
        G_batch, action_batch = self.get_train_batch(states, actions, rewards) 
        history = self.model.fit(states, G_batch, epochs=1, batch_size=len(G_batch), verbose=0)
        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]
        episode_reward = sum(rewards)
        return loss, acc, len(states), episode_reward


    def test(self, env, render=False):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        _, _, rewards = self.generate_episode(env, render=render, train=False)
        episode_reward = sum(rewards)
        return episode_reward, len(rewards)

    def generate_episode(self, env, render=False, train=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        done = False
        state = env.reset()
        while not done:
            # replace random action with policy
            if render:
                env.render()
            action_values =  self.model.predict(np.array(state, ndmin=2)) 
            if train:
                action = self.sample_action(action_values)
            else:
                action = np.argmax(action_values)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
        return np.array(states), np.array(actions), np.array(rewards)

    def save_model(self, suffix):
        self.model.save_weights(os.path.join(OUTPUT_PATH, suffix+'_model.h5'))


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=5e-4, help="Discount.")
    parser.add_argument('--exp', dest='exp', type=str,
                        default="EXP", help="experiment description")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def plot_graph(data, title, xlabel, ylabel):
    plt.figure(figsize=(12,5))
    plt.title(title)
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(OUTPUT_PATH,title+'.png'))

def plot_errorbar(x, y, yerr, title, xlabel, ylabel, label=None):
    plt.figure(figsize=(12,5))
    plt.title(title)
    plt.errorbar(x, y, yerr, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(OUTPUT_PATH, title+'.png'))

def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    test_episodes = 1000
    val_episodes = 100
    lr = args.lr#0.001 #args.lr
    gamma = args.gamma
    render = args.render

    # Create the environment.
    env_name = 'LunarLander-v2'
    #env_name = 'CartPole-v0'
    env = gym.make(env_name)

    # Create the model
    model = create_model()
    model.get_config()
    re = Reinforce(model, lr, gamma, env)
    loss_c = []
    acc_c = []
    rewards_c = []
    val_rewards_c = []
    mean_val_c = []
    std_val_c = []
    suffix = args.exp+'_'+env_name+str(lr)+'_'+str(gamma)
    x = []
    print('Training with lr = ', lr, '| Gamma = ', gamma)
    for i in range(num_episodes):
        [loss, acc, episode_steps, episode_reward] = re.train(env)
        print('TRAINING episode = %d/%d | episode_steps = %d | episode_reward = %d | loss = %f | acc = %f'%(i, num_episodes, episode_steps, episode_reward, loss, acc))
        loss_c.append(loss)
        acc_c.append(acc)
        rewards_c.append(episode_reward)
        if i%500 == 0:
            for j in range(val_episodes):
                
                val_reward, val_steps = re.test(env)        
                val_rewards_c.append(val_reward)

                print('VALIDATION episode = %d/%d | episode_steps = %d | episode_reward = %d '%(j, val_episodes, val_steps, val_reward))
            mean_val_c.append(np.mean(val_rewards_c))
            std_val_c.append(np.std(val_rewards_c))
            x.append(i)

    
    plot_graph(rewards_c, suffix+'_Episode_rewards', 'Episodes', 'Training Rewards')
    plot_graph(loss_c, suffix+'_Training_loss', 'Episodes', 'Training Loss')
    plot_errorbar(x, mean_val_c, std_val_c, suffix+'_mean_val_rewards', 'Episodes', 'Val Rewards', label='std')
    
    #save the mdoel
    re.save_model(suffix)
    # test for a 1000 episodes
    test_rewards = []
    for i in range(test_episodes):
        episode_reward, episode_steps = re.test(env)
        test_rewards.append(episode_reward)
    
        print('TESTING episode = %d/%d | episode_steps = %d | episode_reward = %d '%(i, test_episodes, episode_steps, episode_reward))

    plot_graph(test_rewards, suffix+'_Test_rewards', 'Test Episodes', 'Test Rewards')

if __name__ == '__main__':
    main(sys.argv)
