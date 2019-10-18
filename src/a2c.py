import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization



import gym
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce, ACTION_DIM, STATE_DIM, OUTPUT_PATH, plot_graph, plot_errorbar


def create_actor_model():
    """ This function create the NN using keras library.
    """

    model = Sequential()
    model.add(Dense(16,
                    input_shape=(STATE_DIM,),
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16,
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16,
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(ACTION_DIM,
                    activation='softmax',
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    return model

def create_critic_model():
    """ This function create the NN using keras library.
    """

    model = Sequential()
    model.add(Dense(16,
                    input_shape=(STATE_DIM,),
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16,
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(16,
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    #model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1,
                    kernel_initializer=keras.initializers.VarianceScaling(scale=1.0, distribution='uniform', mode='fan_avg'),
                    bias_initializer=keras.initializers.Zeros()))
    return model

class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, env, model, lr, critic_model, critic_lr, gamma, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n
        self.gamma = gamma
        self.gamma_array = np.array([gamma**i for i in range(n)])
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        self.model.compile(loss='categorical_crossentropy',
                                 optimizer=keras.optimizers.Adam(lr),
                                 metrics=['accuracy'])
        self.critic_model.compile(loss=keras.losses.mean_squared_error,
                                  optimizer=keras.optimizers.Adam(critic_lr),
                                  metrics=['accuracy'])

    def get_RV(self, states, rewards):
        assert len(states) == len(rewards)
        R = np.zeros(len(rewards))
        V = np.zeros(len(states))
        T = len(states)
        zeros_ = np.zeros(self.n - 1)
        appended_rewards = np.concatenate((rewards, zeros_), axis=0)
        for t in range(T-1, -1, -1):
            if t + self.n >= T:
                V[t] = 0
            else:
                V[t] = self.critic_model.predict( np.array(states[t+self.n], ndmin=2) )


            R[t] = self.gamma**(self.n) * V[t] + np.dot( self.gamma_array , appended_rewards[t:t+self.n] )            
            


        return R, V


    def get_train_batch(self, states, actions, rewards):
        R, V = self.get_RV(states, rewards)        
        assert len(R) == len(V)

        diff = R - V
        diff_batch = np.zeros([len(diff), ACTION_DIM])
        diff_batch[np.arange(len(diff)), actions] = diff


        return diff_batch, R


    def train(self, env):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        [states, actions, rewards] = self.generate_episode(env, render=False, train=True)  
        diff_batch, R = self.get_train_batch(states, actions, rewards)
        #[] = self.get_train_batch_critic()
        #  model.fit on both models here
        actor_history = self.model.fit(states, diff_batch, epochs=1, batch_size=len(states), verbose=0)
        actor_loss = actor_history.history['loss'][-1]
        actor_acc = actor_history.history['acc'][-1]

        critic_history = self.critic_model.fit(states, R, epochs=1, batch_size=len(states), verbose=0)
        critic_loss = critic_history.history['loss'][-1]
        critic_acc = critic_history.history['acc'][-1]
        episode_reward = sum(rewards)
 
        # compute episode rewards
        return actor_loss, actor_acc, critic_loss, critic_acc, len(rewards), episode_reward   

    def save_model(self, suffix):
        self.model.save_weights(os.path.join(OUTPUT_PATH, suffix+'actor.h5'))
        self.critic_model.save_weights(os.path.join(OUTPUT_PATH, suffix+'critic.h5'))


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--gamma', dest='gamma', type=float,
                        default=0.99, help="Discount factor.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.add_argument('--exp', dest='exp', type=str,
                        default="EXP", help="experiment description")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    gamma = args.gamma
    test_episodes = 1000
    val_episodes = 100    #render = args.render

    # Create the environment.
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)

    # TODO: Create the model.
    actor_model = create_actor_model()
    critic_model = create_critic_model()

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = A2C(env, actor_model, lr, critic_model, critic_lr, gamma=gamma, n=n)

    actor_loss = []
    critic_loss = []
    rewards_c = []
    val_rewards_c = []
    mean_val_c = []
    std_val_c = []
    x = []    



    suffix = args.exp+'_'+env_name+'actor_lr_'+str(lr)+'critic_lr'+str(critic_lr)+'gamma_'+str(gamma)+'n_'+str(n)
    for i in range(num_episodes):
        [a_l, a_a, c_l, c_a, episode_steps, episode_reward] = a2c.train(env)
        print('TRAINING episode = %d/%d | episode_steps = %d | episode_reward = %d | actor loss = %f | critic loss = %f'%(i, num_episodes, episode_steps, episode_reward, a_l, c_l))
        actor_loss.append(a_l)
        critic_loss.append(c_l)
        rewards_c.append(episode_reward)
        if i%500 ==0:
            for j in range(val_episodes):
                val_reward, val_steps = a2c.test(env)
                val_rewards_c.append(val_reward)
                print('VALIDATION episode = %d/%d | episode_steps = %d | episode_reward = %d '%(j, val_episodes, val_steps, val_reward))
            mean_val_c.append(np.mean(val_rewards_c))
            std_val_c.append(np.std(val_rewards_c))
            x.append(i)
    
        if i%5000==0:
            
            plot_graph(rewards_c, suffix+str(i)+'_TrainingRewards', 'Episodes', 'TrainingRewards')
            plot_graph(actor_loss, suffix+str(i)+'_ActorTrainingLoss', 'Episodes', 'ActorTrainingLoss')
            plot_graph(critic_loss, suffix+str(i)+'_CriticTrainingLoss', 'Episodes', 'CriticTrainingLoss')
        
            plot_errorbar(x, mean_val_c, std_val_c, suffix+str(i)+'_mean_val_rewards', 'Episodes', 'Val Rewards', label='std')
            a2c.save_model(str(i)+'_'+suffix)


    plot_graph(rewards_c, suffix+'Complete_TrainingRewards', 'Episodes', 'TrainingRewards')
    plot_graph(actor_loss, suffix+'Complete_ActorTrainingLoss', 'Episodes', 'ActorTrainingLoss')
    plot_graph(critic_loss, suffix+'Complete_CriticTrainingLoss', 'Episodes', 'CriticTrainingLoss')
        
    plot_errorbar(x, mean_val_c, std_val_c, suffix+'Compelte_mean_val_rewards', 'Episodes', 'Val Rewards', label='std')
    a2c.save_model('Complete_'+suffix)

    # Test episodes
    test_rewards = []
    for i in range(test_episodes):
        episode_reward, episode_steps = a2c.test(env)
        test_rewards.append(episode_reward)
        print('TESTING episode = %d/%d | episode_steps = %d | episode_reward = %d '%(i, test_episodes, episode_steps, episode_reward))

    plot_graph(test_rewards, suffix+'_Test_rewards', 'Test Episodes', 'Test Rewards')
    

    
if __name__ == '__main__':
    main(sys.argv)






























