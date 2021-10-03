#Name: Enrico Calleris

import numpy as np
import random
from tqdm import tqdm
import seaborn as sns
import math
import matplotlib.pyplot as plt

#Code adapted from:
#https://towardsdatascience.com/reinforcement-learning-rl-101-with-python-e1aa0d37d43b

#Initialize Gridworld

reward = -1
gridSize = 9
terminalStates = [[6,5],[gridSize-1,gridSize-1]]
walls = [[1,2],[1,3],[1,4],[1,5],[1,6],[2,6],[3,6],[4,6],[5,6],\
             [1,6],[7,1],[7,2],[7,3],[7,4]]
actions = ((-1,0),(0,1),(1,0),(0,-1))
n = 1000

returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = np.zeros([gridSize*gridSize, len(actions)])
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]
qsa = np.zeros((gridSize, gridSize))
Q = np.zeros((gridSize, gridSize))
deltaqsa = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
pi = {(i, j): {action: 0.25 for action in actions} for i in range(gridSize) \
      for j in range(gridSize)}
deltapi = {(i, j): {action: list() for action in actions} \
           for i in range(gridSize) for j in range(gridSize)}

#Part 1: Monte Carlo Policy Evaluation

def generateEpisode(learning = None, gamma = 0.9, alpha = 0.4):

    initialState = random.choice(states[1:-1])
    episode = []
    while True:
        if list(initialState) in terminalStates or \
           list(initialState) in walls: return episode
        action = random.choice(actions)
        finalState = np.array(initialState)+np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState) \
           or list(finalState) in walls: finalState -= np.array(action)
        if list(finalState) in terminalStates:
            if list(finalState) == terminalStates[0]:
                 episode.append([list(initialState), action, -50, list(finalState)])
            else: episode.append([list(initialState), action, 50, list(finalState)])
        else: episode.append([list(initialState), action, reward, list(finalState)])
        initialState = finalState

        if learning == 'sarsa' and len(episode) > 2:
            s = tuple(episode[-2][0])
            a = episode[-2][1]
            rew = episode[-2][2]
            sPrime = tuple(episode[-2][3])
            aPrime = episode[-1][1]
            piOld = pi[s][a]
            piPrime = pi[sPrime][aPrime]
            piNew = piOld + alpha*(rew+gamma*piPrime - piOld)
            deltapi[s][a].append(np.abs(piOld-piNew))
            pi[s][a] = piNew

        elif learning == 'qlearning' and len(episode)>2:
            s = tuple(episode[-2][0])
            a = tuple(episode[-2][1])
            rew = episode[-2][2]
            sPrime = episode[-2][3]
            aPrime = episode[-1][1]
            piOld = pi[s][a]
            piPrime = max(pi[sPrime].values())
            piNew = piOld + alpha*(rew+gamma*piPrime - piOld)
            deltapi[s][a].append(np.abs(piOld-piNew))
            pi[s][a] = piNew

    return episode

def mcpolicyeval():

    V = np.zeros([gridSize, gridSize])
    gamma = 1

    for it in tqdm(range(n)):
        episode = generateEpisode()
        G = 0
        for j, step in enumerate(episode[::-1]):
            G = gamma*G + step[2]
            if step[0] not in [x[0] for x in episode[::-1][len(episode)-j:]]:
                idx = (step[0][0], step[0][1])
                returns[idx].append(G)
                newValue = np.average(returns[idx])
                deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]]-newValue))
                V[idx[0], idx[1]] = newValue
    return V

#Creation of a Heatmap with visualisation of results of MC Policy Evaluation
#Adapted from: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

def heatmap(v):
    
    v = np.around(np.array(v),0)
    fig, ax = plt.subplots()
    im = ax.imshow(a)
    ax.set_xticks(np.arange(9))
    ax.set_yticks(np.arange(9))
    ax.set_xticklabels(np.arange(1,10))
    ax.set_yticklabels(np.arange(1,10))
    for i in range(9):
        for j in range(9):
            text = ax.text(j, i, a[i, j], ha="center", va="center", color="w")
    ax.set_title('Monte Carlo Policy Evaluation (v)')
    fig.tight_layout()
    plt.show()

#Part 2: SARSA

def sarsa(gamma = 0.9, alpha = 0.4):

    for it in tqdm(range(n)):
        score = 0
        episode = generateEpisode('sarsa')

        for j, step in enumerate(episode[::-1]):            
            score = gamma*score + step[2]
            index = (step[0][0], step[0][1])
            returns[index].append(score)
            q = np.average(returns[index])
            deltaqsa[index[0], index[1]].append(np.abs(Q[index[0], index[1]]-q))
            Q[index[0], index[1]] = q

    return deltaqsa

#Part 3: Q-Learning

def qLearning(gamma = 0.9, alpha = 0.4):
    
    for it in tqdm(range(n)):
        score = 0
        m = 0
        episode = generateEpisode('qlearning')

        for j, step in enumerate(episode[::-1]):          
            score = gamma*score + step[2]
            index = (step[0][0], step[0][1])
            returns[index].append(score)
            q = np.average(returns[index])
            deltaqsa[index[0], index[1]].append(np.abs(Q[index[0], index[1]]-q))
            Q[index[0], index[1]] = q

    return deltaqsa

#Function to plot the results of SARSA and Q-Learning

def graph(file):
    plt.figure(figsize=(20, 10))
    cells = [list(state)[:50] for state in file.values()]
    for cell in cells:
        plt.plot(cell)
    plt.show

#Main Function

def main():

    V = mcpolicyeval()
    heatmap(V)
    delta = sarsa()
    graph(delta)
    #return pi
    #delta = qLearning()
    #print(delta)
    #return pi

main()
