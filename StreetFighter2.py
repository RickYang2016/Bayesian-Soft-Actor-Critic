# -*- coding: utf-8 -*-
#!/usr/bin/python

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions import Categorical
from itertools import count
import torch.optim as optim
import gym, retro, os
import time, random
import numpy as np
from gym import wrappers
from gym.envs.classic_control import rendering
# from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class SF2Discretizer(Discretizer):
    """
    Use Sonic-specific discrete actions
    based on https://github.com/openai/retro-baselines/blob/master/agents/sonic_util.py
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[[],['LEFT'],['RIGHT'],['DOWN'],['UP'],['DOWN','LEFT'],['DOWN','RIGHT'],['UP','LEFT'],['UP','RIGHT'],['A'],['B'],['C'],['X'],['Y'],['Z']])


class SubPolicyNetType1(nn.Module):
    def __init__(self, state_size, action_size):
        super(SubPolicyNetType1,self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # self.conv1 = nn.Conv2d(self.state_size, 32, 8, 4)
        self.conv1 = nn.Conv2d(self.state_size, 32, 3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        # self.hidden1 = nn.Linear(2048, 512)
        # self.hidden2 = nn.Linear(128, 256)
        self.predict = nn.Linear(2048, self.action_size)

    def forward(self, state):
        output = F.relu(self.conv1(state.unsqueeze(0)))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = torch.flatten(output, 1)
        # output = F.relu(self.hidden1(state))
        # output = F.relu(self.hidden1(output))
        output = self.predict(output)
        # out = F.sigmoid(out)
        # distribution = Categorical(F.softmax(output, dim=-1))
        distribution = F.softmax(output, dim=-1)
        # print(distribution)
        return distribution

class SubPolicyNetType2(nn.Module):
    def __init__(self, state_size, action_dist_size, action_size):
        super(SubPolicyNetType2,self).__init__()
        self.state_size = state_size
        self.action_dist_size = action_dist_size
        self.action_size = action_size
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=4, padding=1)
        self.conv1 = nn.Conv2d(self.state_size, 32, 3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.hidden1 = nn.Linear(self.action_dist_size, 128)
        self.hidden2 = nn.Linear(128, 64)
        self.hidden3 = nn.Linear(64, 16)
        self.predict = nn.Linear(2064, self.action_size)

        # self.predict = nn.Linear(2048 + self.action_dist_size, self.action_size)


    def forward(self, state, action_dist):
        output = F.relu(self.conv1(state.unsqueeze(0)))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = torch.flatten(output, 1)
        tmp_output = F.relu(self.hidden1(action_dist))
        tmp_output = F.relu(self.hidden2(tmp_output))
        output = self.predict(torch.cat((output, self.hidden3(tmp_output)), 1))
        distribution = F.softmax(output, dim=-1)

        # output = torch.flatten(output, 1)

        # out = F.sigmoid(out)
        # distribution = Categorical(F.softmax(output, dim=-1))
        # print(distribution)
        return distribution

class Critic(nn.Module):
    def __init__(self, state_size, action_list_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_list_size = action_list_size
        self.conv1 = nn.Conv2d(self.state_size, 32, 3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.hidden1 = nn.Linear(self.action_list_size, 64)
        self.hidden2 = nn.Linear(64, 16)
        self.predict = nn.Linear(2064, 1)

    def forward(self, state, actionList):
        output = F.relu(self.conv1(state.unsqueeze(0)))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = torch.flatten(output, 1)
        tmp_output = F.relu(self.hidden1(actionList))
        value = self.predict(torch.cat((output, self.hidden2(tmp_output)), 1))

        return value

class Critic1(nn.Module):
    def __init__(self, state_size):
        super(Critic1, self).__init__()
        self.state_size = state_size
        self.conv1 = nn.Conv2d(self.state_size, 32, 3, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.predict = nn.Linear(2048, 1)

    def forward(self, state):
        output = F.relu(self.conv1(state.unsqueeze(0)))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = torch.flatten(output, 1)
        value = self.predict(output)

        return value

def set_lags():
    #actions
    neutralL = 0
    leftL = 0
    rightL = 0 
    downL = 0
    upL = 0
    down_leftL = 0
    down_rightL = 0
    up_leftL = 0
    up_rightL = 0
    low_kickL = 10
    medium_kickL = 20
    high_kickL = 30
    low_punchL = 10
    medium_punchL = 20
    high_punchL = 30

    return [neutralL, leftL, rightL, downL, upL, down_leftL, down_rightL, up_leftL, up_rightL, low_kickL, medium_kickL, high_kickL, low_punchL, medium_punchL, high_punchL]


# def ActionOneHot(actionList):
#     length = max(actionList) + 1
#     size = len(actionList)
#     one_hot_encode = torch.zeros(size, length).scatter_(1, torch.tensor(actionList).unsqueeze(1), 1)
#     result = one_hot_encode.view([size, 1, length])
#     result = result.view(-1).unsqueeze(0)

#     return result

# def ActionTensor(actionList, tensor):
#     length = max(actionList) + 1
#     size = len(actionList)
#     one_hot_encode = torch.zeros(size, length).scatter_(1, torch.tensor(actionList).unsqueeze(1), 1)
#     tmpTensor = tensor.reshape([-1, tensor.size()[0]])
#     result = torch.cat((one_hot_encode, tmpTensor), 1)
#     result = result.view([size, 1, length + 1])
#     result = result.view(-1).unsqueeze(0)

#     return result

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

    return result

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []

    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)

    return returns

def NextActionList(state):
    nextActionTemList = [0, 0, 0, 0]

    subDist1 = subPolicy1(state)
    actionDist1 = ActionTensor(actionList1, subDist1)
    nextActionTemList[0] = actionList1[Categorical(subDist1).sample()]

    subDist2 = subPolicy2(state, actionDist1)
    actionDist2 = ActionTensor(actionList2, subDist2)
    nextActionTemList[1] = actionList2[Categorical(subDist2).sample()]

    subDist3 = subPolicy3(state, actionDist2)
    actionDist3 = ActionTensor(actionList3, subDist3)
    nextActionTemList[2] = actionList3[Categorical(subDist3).sample()]

    subDist4 = subPolicy4(state, actionDist3)
    nextActionTemList[3] = actionList4[Categorical(subDist4).sample()]

    return nextActionTemList


def Training(subPolicy1, subPolicy2, subPolicy3, subPolicy4, critic, n_iters):
    global actionTemList

    optimizerP1 = optim.Adam(subPolicy1.parameters(), lr = 0.000001)
    optimizerP2 = optim.Adam(subPolicy2.parameters(), lr = 0.000001)
    optimizerP3 = optim.Adam(subPolicy3.parameters(), lr = 0.000001)
    optimizerP4 = optim.Adam(subPolicy4.parameters(), lr = 0.000001)
    optimizerC = optim.Adam(critic.parameters())

    sleep_seconds = 0.01
    # env.viewer = None
    # env.viewer = rendering.Viewer(600, 400)
    
    score = 0
    action = 0
    count = 4
    inputLag = 0
    old_health = 176
    old_enemy_health = 176
    old_score = 0
    old_enemy_matches_won = 0
    old_matches_won = 0
    sumReward = 0

    for i in range(n_iters):
        state = env.reset()
        log_probs1 = []
        log_probs2 = []
        log_probs3 = []
        log_probs4 = []
        values = []
        rewards = []
        masks = []
        entropy1 = 0
        entropy2 = 0
        entropy3 = 0
        entropy4 = 0
        env.reset()

        # for j in range(10000):
        while True:
            env.render()
            state = torch.FloatTensor(state).to(device)

            # time.sleep(sleep_seconds)

            # if count < 4 and np.array(actionTemList).all() != 0:
            if count < 4:
                if(inputLag > 0):
                    next_state, reward, _, _ = env.step(0)
                    inputLag -= 1
                    continue
                
                inputLag = set_lags()[actionTemList[count]]

                # print(state)

                # next_state, _, done, info = env.step(actionTemList[count])
                next_state, _, _, info = env.step(actionTemList[count])

                done = False

                reward = (old_enemy_health - info['enemy_health']) - (old_health - info['health']) / 50
                # reward = info['score'] - old_score

                # if info['enemy_matches_won'] - old_enemy_matches_won == 1:
                #     sumReward -= 10000
                # elif info['matches_won'] - old_matches_won == 1:
                #     sumReward += 10000

                # old_enemy_matches_won = info['enemy_matches_won']
                # old_matches_won = info['matches_won']

                sumReward += reward

                if info['enemy_matches_won'] == 2:
                    done = True
                    print(done)

                masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

                if count == 3:
                    values.append(value)
                    rewards.append(torch.tensor([sumReward], dtype=torch.float, device=device))
                    log_probs1.append(log_prob1)
                    log_probs2.append(log_prob2)
                    log_probs3.append(log_prob3)
                    log_probs4.append(log_prob4)

                    state = next_state
                    sumReward = 0
                    actionTemList = [0, 0, 0, 0]

                old_score = info['score']
                old_health = info['health']
                old_enemy_health = info['enemy_health']
                state = next_state
                count += 1

                if info['enemy_matches_won'] == 2:
                    env.reset()
                    break
                # print(rewards)
            elif count == 4 and np.array(actionTemList).all() == 0:
                # actionTemList[0] = random.sample(actionList1, 1)[0]
                # actionTemList[1] = random.sample(actionList2, 1)[0]
                # actionTemList[2] = random.sample(actionList3, 1)[0]
                # actionTemList[3] = random.sample(actionList4, 1)[0]

                # building the deep neural network based on the corresponding Bayesian Strategy Net        
                subDist1 = subPolicy1(state)
                actionDist1 = ActionTensor(actionList1, subDist1)
                subAction1 = Categorical(subDist1).sample()
                actionTemList[0] = actionList1[subAction1]

                subDist2 = subPolicy2(state, actionDist1)
                actionDist2 = ActionTensor(actionList2, subDist2)
                subAction2 = Categorical(subDist2).sample()
                actionTemList[1] = actionList2[subAction2]

                subDist3 = subPolicy3(state, actionDist2)
                actionDist3 = ActionTensor(actionList3, subDist3)
                subAction3 = Categorical(subDist3).sample()
                actionTemList[2] = actionList3[subAction3]

                subDist4 = subPolicy4(state, actionDist3)
                subAction4 = Categorical(subDist4).sample()
                actionTemList[3] = actionList4[subAction4]

                actionListEncoding = ActionOneHot(actionTemList)
                value = critic(state, actionListEncoding)


                # subDist1 = subPolicy1(state)
                # # actionDist1 = ActionTensor(actionList1, subDist1)
                # subAction1 = Categorical(subDist1).sample()
                # actionTemList[0] = actionList1[subAction1]

                # subDist2 = subPolicy2(state)
                # # actionDist2 = ActionTensor(actionList2, subDist2)
                # subAction2 = Categorical(subDist2).sample()
                # actionTemList[1] = actionList2[subAction2]

                # subDist3 = subPolicy3(state)
                # # actionDist3 = ActionTensor(actionList3, subDist3)
                # subAction3 = Categorical(subDist3).sample()
                # actionTemList[2] = actionList3[subAction3]

                # subDist4 = subPolicy4(state)
                # subAction4 = Categorical(subDist4).sample()
                # actionTemList[3] = actionList4[subAction4]

                # actionListEncoding = ActionOneHot(actionTemList)
                # value = critic(state)
                
               
                log_prob1 = Categorical(subDist1).log_prob(subAction1).unsqueeze(0)
                entropy1 += Categorical(subDist1).entropy().mean()
                # log_probs1.append(log_prob1)

                
                log_prob2 = Categorical(subDist2).log_prob(subAction2).unsqueeze(0)
                entropy2 += Categorical(subDist2).entropy().mean()
                # log_probs2.append(log_prob2)

                
                log_prob3 = Categorical(subDist3).log_prob(subAction3).unsqueeze(0)
                entropy3 += Categorical(subDist3).entropy().mean()
                # log_probs3.append(log_prob3)

                
                log_prob4 = Categorical(subDist4).log_prob(subAction4).unsqueeze(0)
                entropy4 += Categorical(subDist4).entropy().mean()
                # log_probs4.append(log_prob4)

                print(actionTemList)
                
                count = 0

        next_state = torch.FloatTensor(next_state).to(device)

        nextActionList = NextActionList(next_state)
        next_value = critic(next_state, ActionOneHot(nextActionList))

        # next_value = critic(next_state)

        returns = compute_returns(next_value, rewards, masks)

        log_probs1 = torch.cat(log_probs1)
        log_probs2 = torch.cat(log_probs2)
        log_probs3 = torch.cat(log_probs3)
        log_probs4 = torch.cat(log_probs4)

        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        subActor1_loss = -(log_probs1 * advantage.detach()).mean()
        subActor2_loss = -(log_probs2 * advantage.detach()).mean()
        subActor3_loss = -(log_probs3 * advantage.detach()).mean()
        subActor4_loss = -(log_probs4 * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerP1.zero_grad()
        optimizerP2.zero_grad()
        optimizerP3.zero_grad()
        optimizerP4.zero_grad()
        optimizerC.zero_grad()


        subActor1_loss.backward(retain_graph=True)
        subActor2_loss.backward(retain_graph=True)
        subActor3_loss.backward(retain_graph=True)
        subActor4_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)

        optimizerP1.step()
        optimizerP2.step()
        optimizerP3.step()
        optimizerP4.step()
        optimizerC.step()

    env.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state= 'Champion.Level1.RyuVsGuile.state')
# env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')
# env = DummyVecEnv([lambda: retro.make('StreetFighterIISpecialChampionEdition-Genesis', state = 'RyuVsChunLi', scenario = 'scenario')])
# env = DummyVecEnv([lambda: retro.make('StreetFighterIISpecialChampionEdition-Genesis', state = 'Champion.Level1.RyuVsGuile.state')])

env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis', state= 'Champion.Level1.RyuVsGuile.state')
env = SF2Discretizer(env)

subpolicy1_state_size = env.observation_space.shape[0]
subpolicy2_state_size = env.observation_space.shape[0]
subpolicy3_state_size = env.observation_space.shape[0]
subpolicy4_state_size = env.observation_space.shape[0]
critic_state_size = env.observation_space.shape[0]

actionList1 = [3, 4]
actionList2 = [5, 6]
actionList3 = [1, 2]
actionList4 = [11, 13]
# actionList4 = [9, 10, 11, 12, 13, 14]

# actionList1 = [3]
# actionList2 = [6]
# actionList3 = [2]
# actionList4 = [13]

# actionList1 = [3]
# actionList2 = [5]
# actionList3 = [1]
# actionList4 = [11]

# actionList1 = [1, 2, 3, 7, 8]
# actionList2 = [0, 3, 5, 6]
# actionList3 = [0, 1, 2, 5, 6]
# actionList4 = [0, 9, 10, 11, 12, 13, 14]

# actionList1 = [1, 2, 3]
# actionList2 = [3, 5, 6]
# actionList3 = [1, 2, 5, 6]
# actionList4 = [10, 14]

# actionList1 = [1, 2, 3, 7, 8]
# actionList2 = [3, 5, 6]
# actionList3 = [1, 2, 5, 6]
# actionList4 = [9, 10, 11, 12, 13, 14]

# actionList1 = [1, 2, 3]
# actionList2 = [3, 5, 6]
# actionList3 = [1, 2, 5, 6]
# actionList4 = [9, 10, 11, 12, 13, 14]

# actionList1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# actionList2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# actionList3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# actionList4 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

actionTemList = [0, 0, 0, 0]
actionsList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
length = max(actionsList) + 1
action_one_hot_encoding = torch.zeros(len(actionsList), length).scatter_(1, torch.tensor(actionsList).unsqueeze(1), 1)

# lr = 0.0001



if __name__ == '__main__':
    subPolicy1 = SubPolicyNetType1(subpolicy1_state_size, len(actionList1))
    subPolicy2 = SubPolicyNetType2(subpolicy2_state_size, (length + 1) * len(actionList1), len(actionList2))
    subPolicy3 = SubPolicyNetType2(subpolicy3_state_size, (length + 1) * len(actionList2), len(actionList3))
    subPolicy4 = SubPolicyNetType2(subpolicy4_state_size, (length + 1) * len(actionList3), len(actionList4))
    critic = Critic(critic_state_size, length * len(actionTemList))

    # subPolicy1 = SubPolicyNetType1(subpolicy1_state_size, len(actionList1))
    # subPolicy2 = SubPolicyNetType1(subpolicy2_state_size, len(actionList2))
    # subPolicy3 = SubPolicyNetType1(subpolicy3_state_size, len(actionList3))
    # subPolicy4 = SubPolicyNetType1(subpolicy4_state_size, len(actionList4))
    # critic = Critic1(critic_state_size)

    Training(subPolicy1, subPolicy2, subPolicy3, subPolicy4, critic, n_iters = 1000)