# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 23:47:45 2022

@author: oe21s024
"""

import numpy as np

class Env:
    
    def __init__(self):
        self.height = 5
        self.width = 5
        self.posx   = 0
        self.posy   = 0 
        self.endx = self.width - 1
        self.endy = self.height - 1
        self.action = [0,1,2,3] # left, right, up, down
        self.state_count = self.height*self.width
        self.action_count = len(self.action)
        
        
    def reset(self):
        self.posx = 0
        self.posy = 0
        self.done = False
        return(self.posx, self.posy, self.done)
    
    def step(self, action):
        
        if action == 0: # left 
            self.posx = self.posx - 1 if self.posx > 0 else self.posx  
       
        if action == 1: # right 
            self.posx = self.posx +1 if self.posx < self.width - 1 else self.posx
            
        if action == 2: # up
            self.posy = self.posy -1 if self.posy > 0 else self.posy
            
        if action == 3: # down
            self.posy = self.posy +1 if self.posy < self.height -1 else self.posy
        
        if self.endx == self.posx and self.endy == self.posy:
            done = True
        else: 
            done = False
            
        next_state = self.width * self.posy + self.posx 
        reward = 1 if done else 0
        return(next_state, reward, done)    
    
    def action_choice(self):
        return np.random.choice(self.action)
    
    # diaplaying environment
    def render(self):
        
        for i in range (self.height):
            for j in range(self.width):
                if self.posx == j and self.posy == i :
                    print("O", end= ' ')
                elif self.endx == j and self.endy == i:
                    print("T", end = ' ')
                else:
                    print(".", end = ' ')
            print("")





import os 
import time 


# calling enironment 
env = Env()

# Q table initialization 
q_table = np.random.rand(env.state_count, env.action_count).tolist()


# hyperparameter definition
gamma = 0.1
epochs = 50
epsilon = 0.08
decay = 0.1


# q learning algorithm for traininig 
for i in range(epochs):
    
    # episode rewinding
    state, reward, done = env.reset()
    steps = 0 
    
    # looping 
    while not done:
        # os.system('clear')
        print("epoch # ( ", i+1, "/", epochs, " ) with number of steps :", steps)
        env.render()
        time.sleep(0.05)
        
        # increment steps
        steps += 1
        
        # epsilon greedy algorithm
        
        # exploration strategy 
        if np.random.uniform() < epsilon :
            action = env.action_choice()
        # exploitation strategy
        else:
            action = q_table[state].index(max(q_table[state]))
            
       # taking action
        next_state, reward, done = env.step(action)
       
       # update q table with bellman equation
        q_table[state][action] = reward  + gamma * max(q_table[next_state])
       
       # updating state 
        state = next_state 
    
    if done:
        print("epoch # ( ", i+1, "/", epochs, " ) with number of steps :", steps)
        env.render()
      
    epsilon -= decay * epsilon

    print("\nDone in", steps, "steps".format(steps))
    time.sleep(0.8)
            
        
    
