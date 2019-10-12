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



def create_model():
    """ This function create the NN using keras library.
    """

    model = Sequential()
    model.add(Dense(16,
                    input_shape=(8,),
                    activation='relu',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Ones()))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Ones()))
    model.add(Dense(16,
                    activation='relu',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Ones()))
    model.add(Dense(4,
                    activation='softmax',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Ones()))
    return model

class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model
        self.gamma = 0.9
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr),
                           metrics=['accuracy'])


    def get_G(self, rewards):
        G = np.zeros(len(rewards))
        temp = 0
        for i in range(len(G)-1, -1, -1):
            G[i] = rewards[i] + self.gamma*temp
            temp = G[i]
        return G


    def get_train_batch(self, states, actions, rewards):
        assert len(states) == len(actions)
        action_batch = np.zeros([len(actions), 4])
        action_batch[np.arange(len(actions)), actions] = 1

        G = self.get_G(rewards)
        G_batch = np.zeros([len(G), 4])
        G_batch[np.arange(len(G_batch)), actions] = G

        
  
	
	



        return G_batch, action_batch

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env)
        G_batch, action_batch = self.get_train_batch(states, actions, rewards) 
        history = self.model.fit(states, G_batch, epochs=1, batch_size=len(G_batch), verbose=0)
        loss = history.history['loss'][-1]
        acc = history.history['acc'][-1]
        episode_reward = sum(rewards)
        return loss, acc, len(states), episode_reward


    def test(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        _, _, rewards = self.generate_episode(env)
        episode_reward = sum(rewards)
        return episode_reward, len(rewards)

    def generate_episode(self, env, render=False):
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
            #action = env.action_space.sample()
            action = np.argmax( self.model.predict(np.array(state, ndmin=2)) )
            next_state, reward, done, info = env.step(action)
            #print('state', state, len(state))
            #print('action', action)	
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



def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = 5000#args.num_episodes
    test_episodes = 1000
    lr = 0.001 #args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Create the model
    model = create_model()
    model.get_config()
    re = Reinforce(model, lr)
    loss_c = []
    acc_c = []
    rewards_c = []
    for i in range(num_episodes):
        [loss, acc, episode_steps, episode_reward] = re.train(env)
        print('TRAINING episode = %d/%d | episode_steps = %d | episode_reward = %d | loss = %f | acc = %f'%(i, num_episodes, episode_steps, episode_reward, loss, acc))
        loss_c.append(loss)
        acc_c.append(acc)
        rewards_c.append(episode_reward)

    plot_graph(rewards_c, 'Episdoe_rewards', 'Episodes', 'Training Rewards')
    plot_graph(loss_c, 'Training_loss', 'Episodes', 'Training Loss')

    #save the mdoel
    re.save_model('lunar')

    # test for a 1000 episodes
    test_rewards = []
    for i in range(test_episodes):
        episode_reward, episode_steps = re.test(env)
        test_rewards.append(episode_reward)
    
        print('TESTING episode = %d/%d | episode_steps = %d | episode_reward = %d '%(i, test_episodes, episode_steps, episode_reward))

    plot_graph(test_rewards, 'Test_rewards', 'Test Episodes', 'Test Rewards')

if __name__ == '__main__':
    main(sys.argv)
