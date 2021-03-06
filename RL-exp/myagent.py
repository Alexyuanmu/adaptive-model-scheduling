import gym
from gym import spaces
from gym.utils import seeding

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

import numpy as np
import json
import random
import time
import sys
import pickle
import math

class ExpEnv(gym.Env):
    """
    ExpEnv: Experiment Environment
    Args:
        model_config_json: the configuration file(.json) of models
            data structure:
            {
                "modelname": [
                    FLOAT_VALUE,    # model weight/value/preference
                    ["label_1", "label_2", ...] # label list
                ]
            }
        This file should be edited by the user.

        exec_result_pkl: the execution result file(.pkl) of data
            data structure:
            {
                "data_id": [
                    ("label", conf),    # label & confidence
                    ...
                ]
            }
        This file can be generated by utils/generate_exec_result.py

    """
    def __init__(self, model_config_json, exec_result_pkl):
        # load json data
        self.config_data = json.load(open(model_config_json))
        # init action space
        self.action_list = list(self.config_data.keys())
        self.action_num = len(self.action_list)
        #print("{} models : \n{}\n".format(self.action_num, self.action_list))
        self.action_space = spaces.Discrete(self.action_num)

        # init observation space
        self.label_list = self.merge_label(self.config_data)
        self.label_num = len(self.label_list)
        #print("{} labels : \n{}\n".format(self.label_num, self.label_list))
        self.observation_space = spaces.MultiBinary(self.label_num)

        # for faster mapping from label_name to observation_index
        self.label2idx = self.label_to_idx(self.label_list)

        # load execution result pickle file
        # for simulating the environment feedback
        self.exec_result = pickle.load(open(exec_result_pkl, "rb"))
        self.data_id_list = list(self.exec_result.keys())
        self.data_num = len(self.data_id_list)
        print("{} execution records loaded.".format(self.data_num))

        # init record index
        self.record_idx = 0

    def open_log(self, path):
        self.log = open(path, "w")

    def label_to_idx(self, label_list):
        label2idx = {}
        for idx, labelname in enumerate(label_list):
            label2idx[labelname] = idx
        return label2idx

    def merge_label(self, config_data):
        label_list = []
        for _, l in config_data.values():
            label_list += l
        # remove duplicate labels
        return list(set(label_list))

    def reward(self, N, theta):
        """
        reward = log(theta * N + 1)
        """
        return math.log(theta * N + 1)

    def punish(self):
        """
        punish = -1.0
        """
        return -1.0

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        #print("Current observation:{}".format(self.observation))
        #print("Refers to {}".format(self.label_list))
        #print("{} labels have been obtained.".format(np.sum(self.observation)))

        # based on the selected action
        # compute the reward and move to the next state
        assert self.action_space.contains(action)
        modelname = self.action_list[action]
        self.act_seq.append(modelname)

        #print("Action: #{} : {}".format(action, modelname))
        theta, target_labels = self.config_data[modelname]
        #print("Theta={}".format(theta))

        # compute the reward
        # cur_record = [(label_name, conf), ...]
        cur_record = self.exec_result[self.data_id_list[self.record_idx]]
        add_label_val = 0.
        #print(cur_record)
        modified_idx = []
        for (label_name, conf) in cur_record:
            obs_idx = self.label2idx[label_name]
            # in order to solve the situation that same label have multiple records
            # we do not update the observation now
            # instead, we save the modified_idx
            if (self.observation[obs_idx] == 0) and (label_name in target_labels) :
                #print(label_name, conf)
                add_label_val += conf
                modified_idx.append(obs_idx)
        # update observation
        for obs_idx in modified_idx:
            self.observation[obs_idx] = 1.

        if add_label_val == 0. :
            # punish totally useless action
            r = self.punish()
        else:
            r = self.reward(N=add_label_val, theta=theta)
        #print("Reward={}".format(r))

        done = False
        self.act_count += 1
        if self.act_count == self.action_num:
            done = True

        #print("reward:{}\tcurrent label#:{}\n".format(r, np.sum(self.observation)))
        if done:
            self.log.write("ID={}\tSeq:{}\n".format(self.record_idx, self.act_seq))

        return self.observation, r, done, {}


    def reset(self):
        # reset observation
        self.observation = np.zeros(shape=(self.label_num, ))

        # index of the current data execution result in data_id_list
        self.record_idx = (self.record_idx + 1) % self.data_num

        # the number of selected actions
        self.act_count = 0

        # the selected action sequence
        self.act_seq = []

        return self.observation


class ExpAgent:
    """
    ExpAgent: Experiment RL Agent
    Args:
        weights: (optional) the path to the pretrained weights
        env: the environment that the agent interacts with
    """
    def __init__(self, env, weights=None):
        # init D-QN model
        # based on the environment
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(env.action_space.n))
        model.add(Activation('linear'))
        print(model.summary())

        memory = SequentialMemory(limit=50000, window_length=1)
        self.dqn = DQNAgent(model=model,
            nb_actions=env.action_space.n,
            memory=memory,
            nb_steps_warmup=1000,
            target_model_update=1e-3,
            policy=EpsGreedyQPolicy())
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

        if weights:
            self.dqn.load_weights(weights)

    def save_model(self, path):
        self.dqn.save_weights(filepath=path)
        print("{} saved.".format(path))
