#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 21:51:06 2017

@author: jojo
"""
import numpy as np
import My2048
import random
#import json

move_dict = {
        0:'u',
        1:'d',
        2:'l',
        3:'r'
        }
        
#class definitions, NN neural net and layerconns, layer connections. 

class NN(object):
    
    @classmethod
    def from_file(cls, file_name_lst):
        """
        gets thetas from file to initiate NN object
        """
        thetas = []
        layerLst = []
        try:
            for file in file_name_lst:
                f = open(file, 'r')
                theta = np.fromfile(f)
                thetas.append(theta)
            for theta in thetas:
                layer = np.shape(theta)
                newLayer = LayerConns(layer, theta)
                layerLst.append(newLayer) 
            return NN(layerLst)
            
        except FileNotFoundError:
            print('File not found error')
            return NN()
    
    def __init__(self,layerNos=(16,32,32,4), layerLst=None):
        """
        layerNos is a tuple of layer depths, from input to output
        turn that in to a list of tuples of in and out sizes for layers
        default is 16, 32, 32, 4
        so layers is (16,32), (32,32), (32,4)
        which is the size of the connections in to the next layer, ignoring the 
        bias
        init weights adds bias and provides small random weights with mean 0.
        """
        if layerLst:
            self.layers = layerLst
        else:
            layerLst = []
            layers = list(zip(layerNos, layerNos[1:]))
    
            for layer in layers:
                theta = initWeights(layer)
                newLayer = LayerConns(layer, theta)
    #            print(np.shape(newLayer.theta))
                layerLst.append(newLayer)
            self.layers = layerLst

    def write_thetas(self):
        """
        sends thetas to file after being updated, before agent stopped
        """
        for n,layer in enumerate(self.layers):
            file_name = 'w'+str(n)+'.npy'
            w = open(file_name, 'w')
            layer.theta.tofile(w)
            w.close()

    def get_thetas(self):
        """
        returns list of theta matrices
        """
        thetas = []
        for layer in self.layers:
            thetas.append(layer.theta)
        return thetas
    
    def score_move(self,grid):
        """
        takes a grid, looks at each move, then plays randomly another 10 moves 
        afterwards, and looks for the move with the highest score.
        choses move
        """
        next_grid_list = []
        for move in grid.get_available_moves():
            clone = grid.clone()
            clone.move(move)
            next_grid_list.append((clone, move))
        top_score = -10  
        print(next_grid_list)
        for state, move in next_grid_list:
            score = self.forecast(state)
            if score >= top_score:
                top_score, best_move = score, move
        if not best_move:
            print(grid)
        return best_move
  
    def forecast(self, grid):
        """
        takes a grid, looks ahead 10 random moves, returns a score
        returns -1000000 if can't move, ie game over. 
        """
        score = 0
        for n in range(10):
            if grid.can_move():
                try:
                    grid.get_available_moves()
                    print(grid.get_available_moves())
                    move = random.choice(grid.get_available_moves())
                    print(move)
                    score_inc = grid.move(move)
                    score += score_inc
                    grid.comp_turn()
                except:
                    break

            else:
                score -= 1000
        return score

    def avg_score(self, grid):
        """
        takes grid, looks at average score inc after n moves, returns average score
        """
        n = 5
        scores = []
        for i in range(n):
            if grid.can_move():
                grid.get_available_moves()
                
            
        
            
#    
#    
#    def score(grid):
#        """
#        gets score 
#        """
#        return grid.score
            
        
    def get_y(self,grid):
        """
        scores each move using grid score
        up down left right
        """
        best_move = self.score_move(grid)
        y = [0,0,0,0]
        y[best_move] = 1
        return np.array(y)
    
    def costFnCE(self, output, y, lamb):
        """
        cross entropy cost function
        regularised with lamb for lambda
        """
        m = len(y)

#        summing square thetas
        ssqth = 0
        for layer in self.layers:
            ssqth = ssqth + np.sum(layer.theta[1:,:]**2)
        Jreg = (lamb/(2*m))*ssqth
        yOnes = np.sum(np.log(output)*-y)/m
        yZeros = np.sum(np.log(1-output)*(1-y))/m
        J = yOnes - yZeros + Jreg
        
        return J

        
    def costFn(self,X, y,lamb):
        """
        based on matlab code from ML coursera Ng, 
        pythoned. 
        """
        #cost function part
        X0 = np.append([1],X)
#        a = X
        aS = []
        zS = []
        
        for layer in self.layers:
            a = np.append([1],X)
            z = np.dot(a,layer.theta)      
            X = sigmoid(z)
            aS.append(a)
            zS.append(z)
        output = X
        aS.append(X)
#        print ('len aS',len(aS))
        J = self.costFnCE(output, y, lamb)
        #gradient part
        grads = []
        a = aS.pop()
        z = zS.pop()
        m = np.shape(X0)[0]
        delta = np.subtract(a,y)

        for i in reversed(range(len(self.layers))):
            theta = self.layers[i].theta
            a = aS.pop()  
            Delta = (np.outer(delta.T, a))
            if i != 0:
                z = zS.pop()
                delta = np.multiply(np.dot(theta[1:,:],delta.T).T, \
                                           sigmoidGrad(z))                
            theta_grad = Delta/m
            #regularise theta                
            theta_reg = theta
            theta_reg[:,0] = 0
            theta_reg = np.multiply(theta_reg, lamb/m) 
            theta_grad = theta_grad.T + theta_reg
            grads.append(np.array(theta_grad))
 
        grads.reverse()
#        print('len a',len(a))
        return J, grads
    
    def get_move(self, grid):
        """
        returns move chosen and output, by doing forward pass on neural net
        """
        X = self.grid_input(grid)
        for layer in self.layers:
            a = np.append([1],X)
#            print(np.shape(a))
            z = np.dot(layer.theta.T, a) 
#            print(np.shape(z))
            X = sigmoid(z)
#            print('X',X)
        output = X
        move = np.argmax(output)
        return move, output    

    def grid_input(self, grid):
        """
        takes grid and turns it in to input vector for nn
        """
        input_array = []
        for n in grid.lst:
            if n == 0:
                input_array.append(0)
            else:
                input_array.append(np.log2(n))
        return np.array(input_array)
    
    
    def update_weights(self, grid, lamb, alpha):
        """
        uses cost function to calculate gradients and updates weights accordingly
        """
        X = self.grid_input(grid)
        y = self.get_y(grid)
        J, grads = self.costFn(X, y, lamb)
        theta_flat = roll_out(self.get_thetas())
        grad_flat = roll_out(grads)
        new_theta_flat = theta_flat - alpha*grad_flat
        return new_theta_flat

    def roll_up(self, theta_flat):
        """
        takes flattened thetas and puts them back in their arrays
        then sets new layer thetas
        """
        for layer in self.layers:
            t_shape = layer.theta.shape
            t_size = layer.theta.size
            layer.set_theta(np.reshape(theta_flat[:t_size], t_shape))
            theta_flat = theta_flat[t_size:]
        
    def save_theta(self):
        """
        save weights to use again
        """
        w = open('weights.txt', 'w')
        print(self.get_theta(), file=w)
        
     
     
class LayerConns(object):
    def __init__(self, inOut, theta):
        """
        initialises neural net layer connections, with noIn connections in and noOut connections out
        neuron is binary 
        """
        noIn, noOut = inOut
        self.noIn = noIn
        self.noOut = noOut
        self.theta = theta
        
    def set_theta(self, new_theta):
        self.theta = new_theta

#helper functions    
def roll_out(thetas):
    """
    returns flattened thetas for easier manipulation
    thetas is list of theta arrays
    could also be theta grads
    """
    theta_flat = np.array([])
    for theta in thetas:
#        print(np.shape(theta.flatten()))
        theta_flat = np.append(theta_flat, theta.flatten())
    return theta_flat
    
def flatten(l):

    return [item for sublist in l for item  in sublist]       
    
def initWeights(layer):
    """
    uses np random.rand to return list of 
    random weights between +1 and -1 with a uniform distribution
    layer is a tuple of in and out numbers
    """
    x,y = layer
    return (2*np.random.rand(x+1,y))-1

def sigmoid(n):
    """
    returns the sigmoid fn of n
    """
    return 1/(1+np.exp(-n))
    
def sigmoidGrad(n):
    """
    g = sigmoid(z).*(1-sigmoid(z));
    """
    return np.multiply(sigmoid(n), (1-sigmoid(n)))

if __name__ == '__main__':
    
#    
#    for n in range(10):
#        g = Game(agent='NN')
#        
    g = My2048.Grid()
    #print(g.map)
    g.comp_turn()
    g.comp_turn()
    
    
    nnet = NN()
    #for layer in nnet.layers:
    #    print('theta size', np.shape(layer.theta))
    #    print(layer.theta)
    y = nnet.get_y(g)
    #print(y)
    #
    lamb = 0.01 
    move, output = nnet.get_move(g)
    #print(output)
    #
    X = nnet.grid_input(g)
    J = nnet.costFnCE(output, y, lamb)
    print(J)
    
    thetas = nnet.get_thetas()
    alpha = 0.03
    nnet.update_weights(g, lamb, alpha)
    move, output = nnet.get_move(g)
    
    J = nnet.costFnCE(output, y, lamb)
    
    print(J)


    