import numpy as np
import itertools
from matplotlib import pyplot as plt
# from collections import namedtuple
import random
# from numpy.random import gamma
# from numpy.testing._private.utils import print_assert_equal
# import copy

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T

# from numpy.lib.shape_base import tile

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
        current_tile = self.gameboard.cur_tile_type

        current_tiles = []

        for i in range(len(self.gameboard.tiles)):
            if i == current_tile:
                current_tiles.append(1)
            else:
                current_tiles.append(-1)

        self.current_state[:len(self.gameboard.tiles)] = current_tiles
        self.current_state[len(self.gameboard.tiles):] = current_board

        binary_rep_state = np.where(self.current_state == -1, 0, self.current_state)
        self.state_index = binatodeci(binary_rep_state)

  


    def fn_select_action(self):

        self.action_index = None

        done = False

        move_list = []
        for i in range(len(self.actions)):
            if self.gameboard.fn_move(self.actions[i][0], self.actions[i][1]) == 1:
                move_list.append(0)
            else:
                move_list.append(1)

        if random.randrange(0,1) < self.epsilon:
            self.action_index = random.choice(self.actions, 1, p = move_list)
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

# Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'term'))
# class ReplayMemory(object):

#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def push(self, *args):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = Transition(*args)
#         self.position = (self.position + 1) % self.capacity

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)

# class DQN(nn.Module):

#     def __init__(self, rows, cols, tiles, actions):
#         super(DQN, self).__init__()
#         self.fc1    = nn.Linear(rows * cols + len(tiles) + 1, 64)
#         self.fc2    = nn.Linear(64, 64)
#         self.fc3    = nn.Linear(64, len(actions))

#     def forward(self, data):
#         x = data.float()
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

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

        self.actions            = []
        action_perm             = []
        N_ACTION_ORIENTATIONS   = 4                # tile can rotate to 4 different orientations
        N_ACTION_POSITIONS      = gameboard.N_col  # len(gameboard.N_col) possible positions
        action_perm.append(range(0, N_ACTION_ORIENTATIONS))
        action_perm.append(range(0, N_ACTION_POSITIONS))
        for i in itertools.product(*action_perm):
            self.actions.append(i)      # (action1, action2)                
        self.actions = np.array(self.actions)


        self.Q_net              = DQN(rows=gameboard.N_row, cols=gameboard.N_col, tiles=gameboard.tiles, actions=self.actions)
        #self.Q_net = self.Q_net.float()
        self.Q_target           = DQN(rows=gameboard.N_row, cols=gameboard.N_col, tiles=gameboard.tiles, actions=self.actions)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        self.Q_target.eval()

        self.optimizer = optim.Adam(self.Q_net.parameters(), self.alpha)

        self.replay             = ReplayMemory(capacity=self.replay_buffer_size)

        self.reward_tots        = np.zeros((self.episode_count, ))

        self.sync_count         = 0

    def fn_load_strategy(self,strategy_file):
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file
        

    def fn_read_state(self):

        current_board = self.gameboard.board.flatten()

        tiles = np.zeros((len(self.gameboard.tiles,)), )
        tiles[tiles == 0] = -1
        tiles[self.gameboard.cur_tile_type] = 1

        current_board = np.concatenate((current_board, [self.gameboard.gameover]), axis=None)

        self.current_state_np = np.concatenate((current_board, tiles), axis=None)

        self.current_state = copy.deepcopy(torch.tensor([self.current_state_np]))

    def fn_select_action(self):

        self.action_index = None

        r = np.random.uniform(0, 1)

        
        epsilon_E = max(self.epsilon, 1 - self.episode / self.epsilon_scale)

        if r < epsilon_E:
            self.action_index = np.random.randint(0, len(self.actions))
            self.gameboard.fn_move(self.actions[self.action_index][0], self.actions[self.action_index][1])
        else:
            self.action_index = self.Q_target(self.current_state).argmax().item()
            self.gameboard.fn_move(self.actions[self.action_index][0], self.actions[self.action_index][1])

    def fn_reinforce(self,batch):

        

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
                                                    
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        term_batch = torch.cat(batch.term)

        state_action_values = self.Q_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.Q_target(non_final_next_states).max(1)[0].detach()


        GAMMA = 0.99
        
        expected_state_action_values = reward_batch + next_state_values * GAMMA * term_batch

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.sum(self.reward_tots[range(self.episode-100,self.episode)])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    np.savetxt("Rewards.csv", self.reward_tots)
            if self.episode>=self.episode_count:
                raise SystemExit(0)
                    
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = copy.deepcopy(self.current_state_np)
            
            old_state = torch.tensor([old_state])
            
            #actions = np.zeros((len(self.actions, )))
            #actions[actions == 0] = -1
            #actions[self.action_index] = 1
            action = copy.deepcopy(torch.tensor([[self.action_index]], dtype=torch.long))

            # Drop the tile on the game board
            reward=copy.deepcopy(self.gameboard.fn_drop())

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()

            if self.gameboard.gameover:
                term = torch.tensor([0])
            else:
                term = torch.tensor([1])

            self.replay.push(old_state, action, self.current_state, torch.tensor([reward]), term)

            if len(self.replay) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                transitions = self.replay.sample(self.batch_size)
                batch = Transition(*zip(*transitions))

                self.fn_reinforce(batch)

                if self.sync_count % self.sync_target_episode_count==0:
                    self.Q_target.load_state_dict(self.Q_net.state_dict())
                
                self.sync_count += 1

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
