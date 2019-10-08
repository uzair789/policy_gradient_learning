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

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=keras.optimizers.Adam(lr),
                           metrics=['accuracy'])


    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        return



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
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(state)	
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
        return states, actions, rewards


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


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = 0.001 #args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Create the model
    model = create_model()
    model.get_config()
    re = Reinforce(model, lr)
    for i in range(10):
        s, a, r = re.generate_episode(env, True) 

    # TODO: Create the model.

    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
