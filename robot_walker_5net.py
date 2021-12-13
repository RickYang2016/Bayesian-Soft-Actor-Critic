# -*- coding: utf-8 -*-
#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.distributions import Normal
from itertools import count
import torch.optim as optim
import gym, retro, os, math
import time, random
import numpy as np
from gym import wrappers
from gym.envs.classic_control import rendering
import mujoco_py
from rl_plotter.logger import Logger
# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv


class SubPolicyNetType1(nn.Module):
    def __init__(self, state_size, action_size):
        super(SubPolicyNetType1,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden1 = nn.Linear(self.state_size, 128)
        self.mu_predict = nn.Linear(128, self.action_size)
        self.sigma_predict = nn.Linear(128, self.action_size)

    def Action_selection(self, mu_v, sigma_v):
        action_list = torch.FloatTensor(1, self.action_size)

        for i in range(self.action_size):
            action_list[0][i] = Normal(mu_v[i], sigma_v[i]).sample().clamp(min = -1, max = 1)

        # return action_list.squeeze(0)
        return action_list

    def Auxiliary_net(self, mu_v, sigma_v, action_list):
        p1 = 0
        p2 = 0

        for i in range(action_list.shape[0]):
            # p1 += -((mu_v[i] - action_list[i]) ** 2) / (2 * sigma_v[i].clamp(min=1e-3))
            p1 += -((mu_v[i] - action_list[i]) ** 2) / (2 * sigma_v[i])
            p2 += -torch.log(torch.sqrt(2 * math.pi * sigma_v[i]))

        return (p1 + p2).unsqueeze(0)

    def forward(self, state):
        output = F.relu(self.hidden1(state))
        output_mu = torch.tanh(self.mu_predict(output))
        output_sigma = F.softplus(self.sigma_predict(output))
        action_list = self.Action_selection(output_mu, output_sigma)
        auxiliaryValue = self.Auxiliary_net(output_mu, output_sigma, action_list)

        # print(output_sigma)
        return action_list, auxiliaryValue


class SubPolicyNetType2(nn.Module):
    def __init__(self, state_size, dependence_action_size, action_size):
        super(SubPolicyNetType2,self).__init__()
        self.state_size = state_size
        self.dependence_action_size = dependence_action_size
        self.action_size = action_size
        self.hidden1 = nn.Linear(self.state_size, 128)
        self.hidden2 = nn.Linear(self.dependence_action_size, 128)
        self.hidden3 = nn.Linear(256, 128)
        self.mu_predict = nn.Linear(128, self.action_size)
        self.sigma_predict = nn.Linear(128, self.action_size)

    def Action_selection(self, mu_v, sigma_v):
        action_list = torch.FloatTensor(1, self.action_size)

        for i in range(self.action_size):
            action_list[0][i] = Normal(mu_v[i], sigma_v[i]).sample().clamp(min = -1, max = 1)

        # return action_list.squeeze(0)
        return action_list

    def Auxiliary_net(self, mu_v, sigma_v, action_list):
        p1 = 0
        p2 = 0

        for i in range(action_list.shape[0]):
            # p1 += -((mu_v[i] - action_list[i]) ** 2) / (2 * sigma_v[i].clamp(min=1e-3))
            p1 += -((mu_v[i] - action_list[i]) ** 2) / (2 * sigma_v[i])
            p2 += -torch.log(torch.sqrt(2 * math.pi * sigma_v[i]))

        return (p1 + p2).unsqueeze(0)

    def forward(self, state, action_dist):
        state_output = F.relu(self.hidden1(state))
        dependence_action_output = F.relu(self.hidden2(action_dist))
        output = F.relu(self.hidden3((torch.cat((state_output.unsqueeze(0), dependence_action_output), 1))).squeeze(0))
        output_mu = torch.tanh(self.mu_predict(output))
        output_sigma = F.softplus(self.sigma_predict(output))
        action_list = self.Action_selection(output_mu, output_sigma)
        auxiliaryValue = self.Auxiliary_net(output_mu, output_sigma, action_list)

        return action_list, auxiliaryValue

class Critic(nn.Module):
    def __init__(self, state_size, action_list_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_list_size = action_list_size
        self.hidden1 = nn.Linear(self.state_size, 128)
        self.hidden2 = nn.Linear(self.action_list_size, 128)
        self.predict = nn.Linear(256, 1)

    def forward(self, state, actionList):
        state_output = F.relu(self.hidden1(state))
        action_output = F.relu(self.hidden2(actionList))
        value = self.predict(torch.cat((state_output.unsqueeze(0), action_output), 1))

        return value

def ActionOneHot(actionList):
    size = len(actionList)
    indices = torch.LongTensor(actionList)
    selectActions = torch.index_select(action_one_hot_encoding, 0, indices)
    result = selectActions.view([size, 1, length])
    result = result.view(-1).unsqueeze(0)

    return result

def ActionTensor(actionList, tensor):
    size = len(actionList)
    indices = torch.LongTensor(actionList)
    selectActions = torch.index_select(action_one_hot_encoding, 0, indices)
    tmpTensor = tensor.reshape([-1, tensor.size()[0]])
    result = torch.cat((selectActions, tmpTensor), 1)
    result = result.view([size, 1, length + 1])
    result = result.view(-1).unsqueeze(0)
    # result = result.view(-1).squeeze(0)

    return result

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)

    return returns

def NextActionList(state):
    actionTemList = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    hipActionList, auxiliaryValue1 = subPolicy1(state)
    hipActionEva = ActionTensor(hipList, auxiliaryValue1)
    actionTemList.scatter_(1, torch.tensor([[0, 3]]), hipActionList)

    leftKneeAction, auxiliaryValue2 = subPolicy2(state, hipActionEva)
    leftKneeActionEva = ActionTensor(leftKnee, auxiliaryValue2)
    actionTemList.scatter_(1, torch.tensor([[1]]), leftKneeAction)

    leftAnkleAction, auxiliaryValue3 = subPolicy3(state, leftKneeActionEva)
    actionTemList.scatter_(1, torch.tensor([[2]]), leftAnkleAction)

    rightKneeAction, auxiliaryValue4 = subPolicy4(state, hipActionEva)
    rightKneeActionEva = ActionTensor(rightKnee, auxiliaryValue4)
    actionTemList.scatter_(1, torch.tensor([[4]]), rightKneeAction)

    rightAnkleAction, auxiliaryValue5 = subPolicy5(state, rightKneeActionEva)
    actionTemList.scatter_(1, torch.tensor([[5]]), rightAnkleAction)

    return actionTemList

def Training(subPolicy1, subPolicy2, subPolicy3, subPolicy4, subPolicy5, critic, n_iters):
    optimizerP1 = optim.Adam(subPolicy1.parameters(), lr = 0.00001)
    optimizerP2 = optim.Adam(subPolicy2.parameters(), lr = 0.00001)
    optimizerP3 = optim.Adam(subPolicy3.parameters(), lr = 0.00001)
    optimizerP4 = optim.Adam(subPolicy4.parameters(), lr = 0.00001)
    optimizerP5 = optim.Adam(subPolicy5.parameters(), lr = 0.00001)
    optimizerC = optim.Adam(critic.parameters(), lr = 0.0001)

    sleep_seconds = 0.01
    # env.viewer = None
    # env.viewer = rendering.Viewer(600, 400)


    for i in range(n_iters):
        #Sets an initial state
        state = env.reset()
        log_probs1 = []
        log_probs2 = []
        log_probs3 = []
        log_probs4 = []
        log_probs5 = []
        values = []
        rewards = []
        masks = []
        entropy1 = 0
        entropy2 = 0
        entropy3 = 0
        entropy4 = 0
        entropy5 = 0
        done = False
        count = 0
        env.reset()

        # Rendering our instance 1000 times
        # for j in range(1000):
        while True:
            #renders the environment
            env.render()
            state = torch.FloatTensor(state).to(device)
            count += 1

            actionTemList = torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

            hipActionList, auxiliaryValue1 = subPolicy1(state)
            hipActionEva = ActionTensor(hipList, auxiliaryValue1)
            actionTemList.scatter_(1, torch.tensor([[0, 3]]), hipActionList)

            leftKneeAction, auxiliaryValue2 = subPolicy2(state, hipActionEva)
            leftKneeActionEva = ActionTensor(leftKnee, auxiliaryValue2)
            actionTemList.scatter_(1, torch.tensor([[1]]), leftKneeAction)

            leftAnkleAction, auxiliaryValue3 = subPolicy3(state, leftKneeActionEva)
            actionTemList.scatter_(1, torch.tensor([[2]]), leftAnkleAction)

            rightKneeAction, auxiliaryValue4 = subPolicy4(state, hipActionEva)
            rightKneeActionEva = ActionTensor(rightKnee, auxiliaryValue4)
            actionTemList.scatter_(1, torch.tensor([[4]]), rightKneeAction)

            rightAnkleAction, auxiliaryValue5 = subPolicy5(state, rightKneeActionEva)
            actionTemList.scatter_(1, torch.tensor([[5]]), rightAnkleAction)
            print(actionTemList)
            value = critic(state, ActionTensor(actionsList, actionTemList))

            # [left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
            # next_state, reward, done, info = env.step(np.array([-1, 1, 1, 0.5, 0.1, -0.1]))
            next_state, reward, done, info = env.step(actionTemList)
            # reward += count * 0.1
            print(reward)
            # if j == 999:
            #     done = True

            log_probs1.append(auxiliaryValue1)
            log_probs2.append(auxiliaryValue2)
            log_probs3.append(auxiliaryValue3)
            log_probs4.append(auxiliaryValue4)
            log_probs5.append(auxiliaryValue5)

            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

            state = next_state

            if done:
                env.reset()
                break

        next_state = torch.FloatTensor(next_state).to(device)

        nextActionList = NextActionList(next_state)
        next_value = critic(next_state, ActionTensor(actionsList, nextActionList))

        returns = compute_returns(next_value, rewards, masks)

        log_probs1 = torch.cat(log_probs1)
        log_probs2 = torch.cat(log_probs2)
        log_probs3 = torch.cat(log_probs3)
        log_probs4 = torch.cat(log_probs4)
        log_probs5 = torch.cat(log_probs5)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        subActor1_loss = -(log_probs1 * advantage.detach()).mean()
        subActor2_loss = -(log_probs2 * advantage.detach()).mean()
        subActor3_loss = -(log_probs3 * advantage.detach()).mean()
        subActor4_loss = -(log_probs4 * advantage.detach()).mean()
        subActor5_loss = -(log_probs5 * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        # print(subActor1_loss)

        optimizerP1.zero_grad()
        optimizerP2.zero_grad()
        optimizerP3.zero_grad()
        optimizerP4.zero_grad()
        optimizerP5.zero_grad()
        optimizerC.zero_grad()

        subActor1_loss.backward(retain_graph=True)
        subActor2_loss.backward(retain_graph=True)
        subActor3_loss.backward(retain_graph=True)
        subActor4_loss.backward(retain_graph=True)
        subActor5_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)

        optimizerP1.step()
        optimizerP2.step()
        optimizerP3.step()
        optimizerP4.step()
        optimizerP5.step()
        optimizerC.step()

        # # new scenario modeling saving subPolicy1, subPolicy2, subPolicy3, subPolicy4, and critic
        # torch.save(subPolicy1.state_dict(), '/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_hipPolicy1.pt')
        # torch.save(subPolicy2.state_dict(), '/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_leftKneePolicy2.pt')
        # torch.save(subPolicy3.state_dict(), '/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_leftAnklePolicy3.pt')
        # torch.save(subPolicy4.state_dict(), '/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_rightKneePolicy4.pt')
        # torch.save(subPolicy5.state_dict(), '/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_rightAnklePolicy5.pt')
        # torch.save(critic.state_dict(), '/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_critic.pt')

        logger.update(score = rewards, total_steps = i)

    env.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Setting MountainCar-v0 as the environment
# env = gym.make('HandManipulateBlock-v0')
# env = gym.make('FetchReach-v1')
# env = gym.make('FetchSlide-v1')
# env = gym.make('FetchReach-v1')
# env = gym.make('Humanoid-v2')
# env = gym.make('HumanoidStandup-v2')
env = gym.make('Walker2d-v2')
# env = gym.make('Ant-v2')
# env = gym.make('HalfCheetah-v2')
# env = gym.make('Hopper-v2')
# env = gym.make('Swimmer-v2')
# env = gym.make('Reacher-v2')


subpolicy1_state_size = env.observation_space.shape[0]
subpolicy2_state_size = env.observation_space.shape[0]
subpolicy3_state_size = env.observation_space.shape[0]
critic_state_size = env.observation_space.shape[0]

# # three subnets
# hipList = [0, 3]
# leftLegList = [1, 2]
# rightLegList = [3, 5]

# five subnets
hipList = [0, 3]
leftKnee = [1]
leftAnkle = [2]
rightKnee = [4]
rightAnkleList = [5]

# actionTemList = [0, 0, 0, 0, 0, 0]
actionsList = [0, 1, 2, 3, 4, 5]
# actionsList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

length = max(actionsList) + 1
action_one_hot_encoding = torch.zeros(len(actionsList), length).scatter_(1, torch.tensor(actionsList).unsqueeze(1), 1)
# print(action_one_hot_encoding)


if __name__ == '__main__':
    logger = Logger(log_dir = '/logger/robot_walker', exp_name = 'robot_walker', env_name = 'myenv', seed = 40)

    subPolicy1 = SubPolicyNetType1(subpolicy1_state_size, 2)
    subPolicy2 = SubPolicyNetType2(subpolicy2_state_size, (length + 1) * len(hipList), 1)
    subPolicy3 = SubPolicyNetType2(subpolicy3_state_size, length + 1, 1)
    subPolicy4 = SubPolicyNetType2(subpolicy3_state_size, (length + 1) * len(hipList), 1)
    subPolicy5 = SubPolicyNetType2(subpolicy3_state_size, length + 1, 1)
    critic = Critic(critic_state_size, length * (len(actionsList) + 1))

    # # load temporary trained models playing the game
    # subPolicy1.load_state_dict(torch.load('/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_hipPolicy1.pt'))
    # subPolicy1.eval()
    # subPolicy2.load_state_dict(torch.load('/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_leftKneePolicy2.pt'))
    # subPolicy2.eval()
    # subPolicy3.load_state_dict(torch.load('/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_leftAnklePolicy3.pt'))
    # subPolicy3.eval()
    # subPolicy4.load_state_dict(torch.load('/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_rightKneePolicy4.pt'))
    # subPolicy4.eval()
    # subPolicy5.load_state_dict(torch.load('/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_rightAnklePolicy5.pt'))
    # subPolicy5.eval()
    # critic.load_state_dict(torch.load('/home/herobot/Documents/research/gym/code/robot_walker_trained_models/tmp_critic.pt'))
    # critic.eval()

    # print(subPolicy1)
    # print(subPolicy2)
    # print(subPolicy3)
    # print(critic)

    Training(subPolicy1, subPolicy2, subPolicy3, subPolicy4, subPolicy5, critic, n_iters = 10000)


    