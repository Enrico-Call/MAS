#Name: Enrico Calleris

import numpy as np
import numpy.random as npr
import random
import copy

#Class to create and store the tree

class Tree():

    def __init__(self, depth, c = 2):
        self.depth = depth
        self.node = 0
        self.c = c
        self.max = 0

    def getMax(self):
        return self.max

    def buildTree(self, n = 1):
        if n == 1: name = 'Initial'
        else: name = 'node'+str(self.node); self.updateNode()
        return dict([('name', name),('terminal', False),\
                     ('left', self.buildTree(n+1) if n+1 < self.depth else self.createTerminal()),\
                     ('right', self.buildTree(n+1) if n+1 < self.depth else self.createTerminal()),\
                     ('nvisits', 0), ('val', 0), ('depth', n)])


    def createTerminal(self):
        name = 'node' + str(self.node)
        self.updateNode()
        value = npr.uniform(0,100)
        if self.max < value: self.max = value
        return dict([('name', name),('terminal', True),\
                     ('val', value), ('depth', self.depth)])

    def updateNode(self):
        self.node += 1

    def rollout(self, node, n = 5):
        pos = node
        val = 0
        for i in range(n):
            while pos['terminal'] == False: pos = random.choice([pos['left'], pos['right']])
            if pos['val'] > val: val = pos['val']
        return val

    def mcts(self, parent, n = 50):

        #Selection + Expansion
        
        left = parent['left']
        right = parent['right']

        if left['terminal'] == True and right['terminal'] == True:
                val = max([left['val'], right['val']])
                parent['nvisits'] += 1
                parent['val'] += val 
                return val

        for i in range(n):
            if left['nvisits'] != 0 and right['nvisits'] != 0:
                child = left if ((left['val']/left['nvisits'])+ self.c * \
                                np.sqrt(np.log(parent['nvisits'])/left['nvisits'])) > \
                                ((right['val']/right['nvisits'])+ self.c * \
                                np.sqrt(np.log(parent['nvisits'])/right['nvisits'])) \
                                else right
            else:
                if left['nvisits'] == 0 and right['nvisits'] == 0:
                    child = random.choice([left, right])
                elif left['nvisits'] == 0: child = left
                else: child = right

            #Rollout + Backup

            val = self.rollout(child)
            child['nvisits'] += 1
            child['val'] += val
            parent['nvisits'] += 1
            parent['val'] += val        
        
        val = self.mcts(child)
        return val

#Main Function

def main():

    for i in range(0, 22, 2):
        tree = Tree(12, i/10)
        a = tree.buildTree()
        print(i/10)
        print(tree.getMax())
        diff = np.array([])
        exact = 0
        for j in range(1000):
            b = copy.deepcopy(a)
            diff = np.append(diff, tree.getMax() - tree.mcts(b))
            if diff[-1] == 0: exact += 1
        print(np.mean(diff), np.std(diff), exact)
    
main()

                             
            
            

    
