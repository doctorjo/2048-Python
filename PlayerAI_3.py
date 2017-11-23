#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:15:14 2017

@author: jojo
"""
from BaseAI_3 import BaseAI
import numpy as np
import operator
import math

#import grid_3
#player AI
actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}
class PlayerAI(BaseAI):
    
    def setDepthLimit(self, depthLimit):
        """
        """
        self.depthLimit = depthLimit
        
    def setWeights(self, weights):
        self.weights = weights
        
    def getMove(self, grid):
        #start by using random moves. h = score, d = depth, m= move index
        search = MinimaxAlphaBeta(grid)
        gridTuple = 0, 0, None,-np.inf, np.inf, grid
#        search = Minimax(grid)
        search.setDepthLimit(8)
        search.setDepth(0)
        search.setWeights((0.9, 0.1, 0.7, 0.1))
#        search.setWeights((1, 0.5, 0.1, 0.5, 1))
#        search.setScoreFn(self.scoreFn)
#        gridTuple = 0, 0, None, grid
        move, h = search.minimaxAgent(gridTuple)
#        moves = search.showOptions(grid)
#        print(search.showOptions(grid))
#        print(h, move)
        return move
    
    
def ifZero(lst):
    newLst = []
    for i in range(len(lst)):
        
        if lst[i] !=0:
            newLst.append(lst[i])
        
    return newLst    

class Search(object):

    def __init__(self, grid):
        self.grid = grid
        self.explored = ([])
        self.in_queue = ([])
        self.queue = []
        
        
    def setDepthLimit(self, limit):
        """
        sets depth limit
        """
        self.depthLimit = limit
        
    def setWeights(self, weights):
        """
        sets weights
        """
        self.weights = weights
        
    def setDepth(self, depth):
        """
        sets current depth of search
        """
        self.depth = depth
    
    def scoreFn(self, grid):
        """
        provides heuristic, 
        best weights values so far 45, from (0.6, 0.4, 0.0)
        """
        wSpaces, wMon3, wSmooth, wEdge= self.weights
#        spaces = 32 - 32/(len(grid.getAvailableCells())+0.1)
        biggestVal = ((int(grid.getMaxTile()).bit_length())**2)/10
#        snake = self.snake(grid)
#        mUL = self.monotonicUpperLeft(grid)
#        mAR = self.monotonicAllRound(grid)
        mon3 = self.monotonic3(grid)
        if not grid.canMove():
            return -1000000
        if grid.getMaxTile() == 2048:
            return 1000000
        edgeMax = self.edgeMax(grid)
#        score += self.betterMonotonic(grid)
#        print(spaces, biggestVal, score)
#        sWW = self.snakeWithWeights(grid)
        smooth = self.smoothness(grid)
#        sumArMax = self.sumAroundMax(grid)
#        print ('spaces', wSpaces*len(grid.getAvailableCells()), 'monotonic', wMon3*mon3, 'smooth', smooth*wSmooth, 'edgeMax',edgeMax*wEdge , 'biggestVal',biggestVal)
        return  wSpaces*len(grid.getAvailableCells())+ wMon3*mon3 + \
                    smooth*wSmooth + edgeMax*wEdge + biggestVal#+ cornermax* wCorner#+ sWW*wSWW #+sWW #+ + wmUL*mUL +snake*wSnake +wSpaces*spaces + wBiggestVal*biggestVal + mAR*wmAR
    
    def monotonic(self, grid):
        """
        returns higher score if numbers increase from 
        left to right and from up to down
        """
        score = 0
        c1, c2, c3, c4 = [], [], [], []
        for r in grid.map:
            r0 = ifZero(r)
    #        print(r, r0)
            score += abs(sum((int(x).bit_length()-int(y).bit_length()) for x, y in zip(r0, r0[1:])))
            
            c1.append(r[0])
            c2.append(r[1])
            c3.append(r[2])
            c4.append(r[3])
            
        for c in [c1, c2, c3, c4]:
            c0 = ifZero(c)
    #        print(c, c0)
            score += abs(sum((int(x).bit_length()-int(y).bit_length()) for x, y in zip(c0, c0[1:])))
        
        return score
    
    def edgeMax(self, grid):
        n = self.max_tile_pos(grid)
        x = n//4
        y = n%4
        edgeMax = 0
        if (x == 0 or y == 0) or (x == 3 or y == 3):
            edgeMax = 5
        if ((x ==0 and y == 0) or (x == 3 and y == 0)) \
            or ((x == 0 and y == 3) or (x == 3 and y == 3)):
                edgeMax = 50
                
        return edgeMax 
   
    def monotonic2(self, grid):
        """
        returns higher score if more rows and columns are monotonic
        """
        score = 0
        c1, c2, c3, c4 = [], [], [], []
        for r in grid.map:
            r0 = ifZero(r)
    #        print(r, r0)
            if r0 == []:
                continue
            if is_monotonic(r0):
                for v in r0:
                    score+=int(v).bit_length()-1
            c1.append(r[0])
            c2.append(r[1])
            c3.append(r[2])
            c4.append(r[3])
            
        for c in [c1, c2, c3, c4]:
            c0 = ifZero(c)
            if c0 == []:
                continue
    #        print(c, c0)
            if is_monotonic(c0):
                for v in r0:
                    score+=int(v).bit_length()-1
        
        return score
    
    def monotonic3(self, grid):
        """
        returns higher score if more rows and columns are monotonic
        """
        grid_array = np.reshape(grid.lst, (grid.grid_size, grid.grid_size))
        score = 0
        c1, c2, c3, c4 = [], [], [], []
        for r in grid_array:
            r0 = ifZero(r)
            if r0 == []:
                continue
            if is_monotonic(r0):
                for v in r0:
                    score+=(int(v).bit_length()-1)**2
            c1.append(r[0])
            c2.append(r[1])
            c3.append(r[2])
            c4.append(r[3])
            
        for c in [c1, c2, c3, c4]:
            c0 = ifZero(c)
            if c0 == []:
                continue
    #        print(c, c0)
            if is_monotonic(c0):
                for v in r0:
                    score+=(int(v).bit_length()-1)**2
        
        return score
    
    def smoothness(self, grid):
        """
        returns higher score if lower log difference between adjacent squares
    
        """
        grid_array = np.reshape(grid.lst, (grid.grid_size, grid.grid_size))
        score = 0
        c1, c2, c3, c4 = [], [], [], []
        for r in grid_array:
            r0 = ifZero(r)
            score += smooth(r0)
            
            c1.append(r[0])
            c2.append(r[1])
            c3.append(r[2])
            c4.append(r[3])
            
        for c in [c1, c2, c3, c4]:
            c0 = ifZero(c)
            score+= smooth(c0)
        return 1/1+score

    def snake(self, grid):
        """
        measures snake pattern, ie limits differences between adjacent numbers.
        from http://gjdanis.github.io/2015/09/27/2048-game-and-ai/
        """
        if not grid.canMove():
            return -1000000
    
        snake = []
        for i, col in enumerate(zip(*grid.map)):
            snake.extend(reversed(col) if i % 2 == 0 else col)

        m = max(snake)
        return sum(x/10**n for n, x in enumerate(snake)) - math.pow((grid.map[3][0] != m)*abs(grid.map[3][0] - m), 2)
    
    def monotonicUpperLeft(self, grid):
        """
        from https://codemyroad.wordpress.com/2014/05/14/2048-ai-the-intelligent-bot/
        """
        score = 0
        if not grid.canMove():
            return -1000000
        weights = [[.7,.6,.5,.4], [.6,.5,.4,.3], [.5,.4,.3,.2], [.4,.3,.2,.1]]
        split = zip(weights, grid.map)
        for twoRs in split:
            r = list(map(operator.mul, twoRs[0],twoRs[1]))
            score += sum(r)
            
        return score
    
    def snakeWithWeights(self, grid):
        """
        using principle of weights from https://codemyroad.wordpress.com/2014/05/14/2048-ai-the-intelligent-bot/
        but coding in snake formation, so keeping all in last line.
        with reflection and rotation
        """
        scores = []
        if not grid.canMove():
            return -1000000
        weights = [[.15,.13,.12,.11], [.06,.07,.08,.09], [.02,.01,.005,.004], [.003,.003,.002,.001]]
        for r in range(4):
            for t in range(r):
                weights = rotate(weights)
            score = 0
            split = zip(weights, grid.map)
            for twoRs in split:
                r = list(map(operator.mul, twoRs[0],twoRs[1]))
                score += sum(r)
            scores.append(score)
        weights = reflect(weights)
       #reflect then rotate around again
        for r in range(4):
            for t in range(r):
                weights = rotate(weights)
            score = 0
            split = zip(weights, grid.map)
            for twoRs in split:
                r = list(map(operator.mul, twoRs[0],twoRs[1]))
                score += sum(r)
            scores.append(score)
            #return max of possible scores
        return max(scores)
        
            
    def monotonicAllRound(self, grid):
        """
        as above, from https://codemyroad.wordpress.com/2014/05/14/2048-ai-the-intelligent-bot/
        not just upper left now, checks rotations and transpositions
        
        """
        if not grid.canMove():
            return -1000000

        scores = []
        weights = [[.7,.6,.5,.4], [.6,.5,.4,.3], [.5,.4,.3,.2], [.4,.3,.2,.1]]
        for r in range(4):
            score = 0
            for t in range(r):
                weights = rotate(weights)
            split = zip(weights, grid.map)
            for twoRs in split:
                r = list(map(operator.mul, twoRs[0],twoRs[1]))
                score += sum(r)
            scores.append(score)
            
        return max(scores)
    
    def max_tile_pos(self, grid):
        """
        to work with my grid object
        """
        return np.argmax(grid.lst)
        
#    
#    def getMaxTilePos(self, grid):
#        maxTile = 0
#        for x in range(grid.size):
#            for y in range(grid.size):
#                tile = grid.map[x][y]
#                if maxTile < tile:
#                    pos = (x,y)
#                    maxTile = tile
#    
#        return pos

    def sumAroundMax(self, grid):
        """
        returns sum of squares surrounding the biggest tile
        """
        x,y = self.max_tile_pos(grid)
        possiblePos = [((x+1), y), ((x-1), y), (x, (y+1)), (x,(y-1))]
        tot = 0
        for p,q in possiblePos:
            
            try: 
                tot += grid.map[p][q]
            except:
                continue
        return tot
                
                     
    
    def childrenOfAI(self, gridTuple):
        """
        returns list of child states available from grid
        grid is tuple of stuff, h, d, m, grid
        h = heuristic score
        d = depth
        m = move tuple
        """
        childList = []
        h, d, m, grid = gridTuple
#        print(gridTuple)
#        print('in children of AI')
#        print(gridTuple)
        for move in grid.getAvailableMoves():
            clone = grid.clone()
            clone.move(move)
            h = self.scoreFn(clone)
            childList.append((h, d+1, move, clone))
            
        return childList

    def childrenOfC(self, gridTuple):
        """
        returns list of possible states after computers move
        assumes a 2 is produced, reduces branching.
        grid is tuple of stuff, h, d, m, grid
        does not change depth
        cells is the coord of the cell where the computer puts the random 2
        
        """
        childList = []
        h, d, m, grid = gridTuple
        for cells in grid.getAvailableCells():
#            print('in children of C')
            clone = grid.clone()
            clone.setCellValue(cells, 2)
            h = self.scoreFn(clone)
            m = cells
            childList.append((h, d, m, clone))
#            print(h, d, m, clone)
        return childList
            
    def nextMove(self, grid):
        """
        returns next move
        """
        pass
            
    def showOptions(self, grid):
        """
        shows four move options with score
        for debugging
        """
        moveScore = []
        for move in grid.getAvailableMoves():
            clone = grid.clone()
            clone.move(move)

        return moveScore

#helper functions for the heuristics..

def rotate(weights):
    """
    weights is list of lists, usually a 4x4 grid
    rotate rotates them anticlockwise
    """
    size = len(weights)
    rotated = [[0] * size for i in range(size)]
    for i in range(size):
        for j in range(size):
            rotated[i][j] = weights[j][(size-1)-i]
    return rotated

def reflect(weights):
    """
    weights is list of lists, usually a 4x4 grid
    reflect reflects them in x axis
    """
    size = len(weights)
    reflected = [[0] * size for i in range(size)]
    for i in range(size):
        for j in range(size):
            reflected[i][j]=weights[(size-1)-i][j]
    return reflected
            
def is_monotonic(lst):
    """
    returns true if monotonic
    """
    if len(lst) == 1 or len(lst) == 0:
        return True
    op = operator.le
    if not op(lst[0], lst[-1]):
        op = operator.ge
    return all(op(x,y) for x, y in zip(lst, lst[1:]))    

    
def smooth(lst):
    """
    takes in lst of numbers, 
    returns the sum of the log base 2 differences between each adjacent square
    """
#    print(lst)
    return 1/(1+sum(abs(math.log2(x)-math.log2(y)) for x, y in zip(lst, lst[1:])) )
#    return 1/(1+sum(abs(x.bit_length()-y.bit_length()) for x, y in zip(lst, lst[1:])) )

def smooth2(lst):
    """
    takes in lst of numbers,
    returns the number of times the log base 2 difference is 1 or 0
    """
    pass
            
    
    
class Minimax(Search):
    def __init__(self, grid):
        Search.__init__(self, grid)
#        self.depth = 0
        

    
    def minAgent(self, gridTuple):
        """
        min agent, computer i guess
        """
        h, d, m, grid = gridTuple
#        print(grid)
        cells = grid.getAvailableCells()
        if not cells:
            return (h, d, m, grid)        
        bestScore = float('inf')
        for childTuple in self.childrenOfC(gridTuple):
            h,d,m,child = self.maxAgent(childTuple)
            if h <bestScore:
                best_child = childTuple
                bestScore = h
        return best_child
    
    def maxAgent(self, gridTuple):
         """
         max agent computer player AI
         gridTuple is h, d, m, grid
         """
         h, d, m, grid = gridTuple
         if self.depth > self.depthLimit:
             return (h, d, m, grid)
         self.depth += 1
         if not grid.canMove():
             return (h, d, m, grid)
         bestScore = -float('inf')       
         for childTuple in self.childrenOfAI(gridTuple):
#             print('in maxagent')
#             print(childTuple)
            
             h,d,m,child = self.minAgent(childTuple)

             if h > bestScore:
                 best_child = childTuple
                 bestScore = h
         return best_child
         
    def minimaxAgent(self, gridTuple):
#        print('in minimax agent')
#        print(gridTuple)
        h, d, m, grid = self.maxAgent(gridTuple)
        return m, h
    
        

class MinimaxAlphaBeta(Minimax):
    def __init__(self, grid):
        Minimax.__init__(self, grid)
#        self.alpha = -np.inf
#        self.beta = np.inf
        
    def childrenOfAI(self, gridTuple):
        """
        returns list of child states available from grid
        grid is tuple of stuff, h, d, m, grid
        h = heuristic score
        d = depth
        m = move tuple
        """
        childList = []
        h, d, m, a,b,grid = gridTuple
#        print(gridTuple)
#        print('in children of AI')
#        print(gridTuple)
        for move in grid.getAvailableMoves():
            clone = grid.clone()
            clone.move(move)
            h = self.scoreFn(clone)
            childList.append((h, d+1, move, a,b,clone))
            
        return childList

    def childrenOfC(self, gridTuple):
        """
        returns list of possible states after computers move
        assumes a 2 is produced, reduces branching.
        grid is tuple of stuff, h, d, m, grid
        does not change depth
        cells is the coord of the cell where the computer puts the random 2
        
        """
        childList = []
        h, d, m, a,b, grid = gridTuple
        for cells in grid.getAvailableCells():
#            print('in children of C')
            clone = grid.clone()
            clone.setCellValue(cells, 2)
            h = self.scoreFn(clone)
            m = cells
            childList.append((h, d, m, a,b,clone))
#            print(h, d, m, clone)
        return childList    
    def minimaxAgent(self, gridTuple):
#        print('in minimax agent')
#        print(gridTuple)
        h, d, m, a, b, grid = self.maxAgent(gridTuple)
        return m, h
    
    def minAgent(self, gridTuple):
        """
        min agent, computer i guess
        """
        h, d, m,a,b, grid = gridTuple
#        print(grid)
        cells = grid.getAvailableCells()
        if not cells:
            return (h, d, m,a,b, grid)        
        bestScore = float('inf')
        for childTuple in self.childrenOfC(gridTuple):
#            print('in minAgent, about to go in to max agent')
#            print(self.alpha, self.beta)
            h,d,m,a,b,child = self.maxAgent(childTuple)
            
            if h <bestScore:
                best_child = childTuple
                bestScore = h
            if h <= a:
                break
            if h < b:
                 b = h
#            print('in minAgent')
#            print(h)
#            print(self.alpha, self.beta)
        return best_child   
    
    def maxAgent(self, gridTuple):
         """
         max agent computer player AI
         gridTuple is h, d, m, grid
         """
         h, d, m, a,b,grid = gridTuple
         if self.depth > self.depthLimit:
             return (h, d, m, a,b,grid)
         self.depth += 1
         if not grid.canMove():
             return (h, d, m, a,b,grid)
         bestScore = -float('inf')       
         for childTuple in self.childrenOfAI(gridTuple):
#             print('in max agent, about to go in to min agent')
#             print(childTuple)

             h,d,m,a,b,child = self.minAgent(childTuple)

             if h > bestScore:
                 best_child = childTuple
                 bestScore = h

             if h >= b:
                 break
             if h > a:
                 a = h
#             print(h)
#             print('in max agent')
#             print(self.alpha, self.beta)

         return best_child  
     
def convert(n):
    """
    takes list index and returns corresponding i,j row, column reference
    """
    x = n%4
    y = n//4
    return x,y

#print(convert(4))
    