import numpy as np
import random
import math
import h5py
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def binatodeci(binary):
    deci = sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))
    return int(deci)

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard

        self.actions = []
        for i in range(4):
            for j in range(gameboard.N_col):
                self.actions.append([i,j])
        
        self.current_state = np.zeros((self.gameboard.N_row * self.gameboard.N_col + len(gameboard.tiles), ))
        self.Q_table = np.zeros((2**(self.gameboard.N_row * self.gameboard.N_col + len(gameboard.tiles)), len(self.actions)))
        self.reward_table = np.zeros((self.episode_count, ))

    def fn_load_strategy(self,strategy_file):
        self.Q_table = strategy_file

    def fn_read_state(self):
        current_board = np.ndarray.flatten(self.gameboard.board)
        current_tiles = np.zeros((len(self.gameboard.tiles),))-1
        current_tiles[self.gameboard.cur_tile_type] = 1


        self.current_state[:len(self.gameboard.tiles)] = current_tiles
        self.current_state[len(self.gameboard.tiles):] = current_board

        binary_rep_state = np.where(self.current_state == -1, 0, self.current_state)
        self.state_index = binatodeci(binary_rep_state)

  


    def fn_select_action(self):

        move_list = []
        for i in range(len(self.actions)):
            if self.gameboard.fn_move(self.actions[i][0], self.actions[i][1]) == 1:
                move_list.append(0)
            else:
                move_list.append(1)
        move_list = np.divide(move_list,np.sum(move_list))

        if random.uniform(0,1) < self.epsilon:
            self.action_index = np.random.choice(range(len(self.actions)), 1, p = move_list)
        else:
            while self.gameboard.fn_move(self.actions[np.argmax(self.Q_table[self.state_index, :])][0], self.actions[np.argmax(self.Q_table[self.state_index, :])][1]) == 1:
                self.Q_table[self.state_index, np.argmax(self.Q_table[self.state_index, :])] = - np.inf
            self.action_index = np.argmax(self.Q_table[self.state_index, :])
                  
    
    def fn_reinforce(self,old_state,reward):

        old_action = self.action_index

        self.Q_table[old_state, old_action] = self.Q_table[old_state, old_action] + self.alpha * (reward + np.max(self.Q_table[self.state_index, :]) - self.Q_table[old_state, old_action])

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(round(np.sum(self.reward_table[range(self.episode-100,self.episode)] / 100), 2)),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
                    #np.savetxt("Q_table_episode_" + str(self.episode) + ".csv", self.Q_table, delimiter=",")
                    np.savetxt("Rewards.csv", self.reward_table)
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()


            old_state = self.state_index

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later

            self.reward_table[self.episode] += reward


            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state,reward)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward','term'))


class ReplayMemory(object):

    def __init__(self, capacity):
        # self.memory = deque([],maxlen=capacity)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # """Save a transition"""
        # self.memory.append(Transition(*args))
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, rows, columns, tiles, actions, numberOfNeurons):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(rows*columns + len(tiles), numberOfNeurons)
        self.layer2 = nn.Linear(numberOfNeurons, numberOfNeurons)
        self.layer3 = nn.Linear(numberOfNeurons, len(actions))

    def forward(self, x):
        x = x.float()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class TDQNAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.actions = []
        
        for i in range(4):
            for j in range(gameboard.N_col):
                self.actions.append([i,j])
        self.actions = np.array(self.actions)
        
        self.network = DQN(gameboard.N_row,gameboard.N_col, gameboard.tiles, self.actions, 64)
        self.target_network = DQN(gameboard.N_row,gameboard.N_col, gameboard.tiles, self.actions, 64)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        #self.optimizer = optim.RMSprop(self.network.parameters())
        
        self.optimizer = optim.Adam(self.network.parameters(), self.alpha)
        self.memory = ReplayMemory(self.replay_buffer_size)
        self.steps_done = 0
        self.reward_table = np.zeros((self.episode_count, ))
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        current_board = np.ndarray.flatten(self.gameboard.board)
        
        current_tiles = np.zeros((len(self.gameboard.tiles),))-1
        current_tiles[self.gameboard.cur_tile_type] = 1

        self.current_state_copy = np.zeros((len(current_board)+len(current_tiles),))
        self.current_state_copy[:len(self.gameboard.tiles)] = current_tiles
        self.current_state_copy[len(self.gameboard.tiles):] = current_board
        self.current_state_copy = torch.from_numpy(self.current_state_copy)

        self.current_state = copy.deepcopy(self.current_state_copy)
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self

        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def fn_select_action(self):

        if random.uniform(0,1) < max(self.epsilon,1-self.episode_count/self.epsilon_scale):
            self.action_index = np.random.randint(0,len(self.actions))
            self.gameboard.fn_move(self.actions[self.action_index][0],self.actions[self.action_index][1])
        else:
            self.action_index = self.target_network(self.current_state).argmax().item()
            self.gameboard.fn_move(self.actions[self.action_index][0],self.actions[self.action_index][1])
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        GAMMA = 0.99
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        term_batch = torch.cat(batch.term)

        state_action_values = self.network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_table[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    np.savetxt("Rewards.csv", self.reward_table)
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.memory) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    self.target_network = copy.deepcopy(self.network)
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = copy.deepcopy(self.current_state_copy)
            # Drop the tile on the game board
            reward=copy.deepcopy(self.gameboard.fn_drop())
            action = copy.deepcopy(torch.tensor([[self.action_index]], dtype = torch.long))
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_table[self.episode] += reward
            # Read the new state
            self.fn_read_state()
            if self.gameboard.gameover:
                term = torch.tensor([0])
            else:
                term = torch.tensor([1])
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            self.memory.push(old_state, action, self.current_state, torch.tensor([reward]),term)
            if len(self.memory) >= self.replay_buffer_size:
                transitions = self.memory.sample(self.batch_size)
                batch = Transition(*zip(*transitions))
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                self.fn_reinforce(batch)
                if self.steps_done % self.sync_target_episode_count==0:
                    self.target_network.load_state_dict(self.network.state_dict())
                self.steps_done += 1



class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()