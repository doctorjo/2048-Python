#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:02:52 2017

@author: jojo
"""
import numpy as np
import random
import copy
import PlayerAI_3
import NN2048
import tkinter as tk
#for unix display
colorMap = {
    0 	  : 97 ,
    2     : 32 ,
    4     : 100,
    8     : 34 ,
    16    : 107,
    32    : 46 ,
    64    : 106,
    128   : 44 ,
    256   : 104,
    512   : 42 ,
    1024  : 102,
    2048  : 43 ,
    4096  : 103,
    8192  : 45 ,
    16384 : 105,
    32768 : 41 ,
    65536 : 101,
}
#for tkinter
colour_map = {
    0 	  : "white" ,
    2     : "red",
    4     : "orange red",
    8     : "orange",
    16    : "gold",
    32    : "yellow" ,
    64    : "green yellow",
    128   : "green",
    256   : "turquoise",
    512   : "cyan" ,
    1024  : "cornflower blue",
    2048  : "purple" ,
    4096  : "magneta",
    8192  : "red" ,
    16384 : "orange red",
    32768 : "orange" ,
    65536 : "gold",}

cTemp = "\x1b[%dm%7s\x1b[0m "
human_move_dict = {'w' : 0, 's':1, 'a':2, 'd':3}

move_dict = {
        0:'u',
        1:'d',
        2:'l',
        3:'r'
        }

#python implimentation of 2048


#visualising the game.. use tkinter?
#class 2048App()


#tk.Label(root,)



def move(dir):
    print("will try to move:", dir)
    
#master = tk.Tk()

#a game manager class, sets up game object, sets up player, provides scores

class Game(object):
    def __init__(self, agent='AI', show='on'):
        """
        initialises game, sets up player (human, AI agent, etc), sets up score
        """
        self.grid = Grid()
        self.score = 0
        self.show = show
        if agent == 'AI':
            self.agent = Agent()
        elif agent == 'RA':
            self.agent = RandomAgent()
        elif agent == 'MM':
            self.agent = MiniMaxAgent()
        elif agent == 'NN':
            self.agent = NNAgent()
        else:
            self.agent = Human()
            
    def start(self):
        """
        starts game, uses grid object already made by game manager
        """
        self.grid.start()
        if self.show =='on':
            print(self.grid.map)
            self.display_grid_unix()
        while self.grid.can_move():
            moved = False
            while not moved:
                
                move = self.agent.get_move(self.grid)
                if self.grid.can_move_dir(move):
                    moved = True
                else:
                    continue
            score_inc = self.grid.move(move)
            self.score = self.score + score_inc
            self.grid.comp_turn()
            
            if self.show=='on':
                self.display_grid_unix()
                print(self.score)
        if self.show == 'on':
            print ('game over')
        return self.score, self.grid.get_max_tile()
    
    def display_grid_first(self):
        """
        trying to use tkinter to display game
        """
        root = tk.Tk()
    
        tk.Label(root, text="2048", bg="red", fg="white", padx=1).grid(row=0, column = 2, columnspan = 6)
        U = tk.Button()
        U["text"] = "Up"
        U["command"] = move("Up")
        U.grid(row=1, rowspan=4, column = 2, columnspan = 6)
        
        L = tk.Button()
        L["text"] = "Left"
        L["command"] = move("Left")
        L.grid(row=4, column=0)
        
        R = tk.Button()
        R["text"] = "Right"
        R["command"] = move("Right")
        R.grid(row=4,rowspan=4, column=6)
        
        D = tk.Button()
        D["text"]="Down"
        D["command"]=move("Down")
        D.grid(row=6,column = 2, columnspan=6)
        
        quit = tk.Button(text="QUIT", fg="red",
                              command=root.destroy)
        quit.grid(row=7, column = 2,columnspan=6)
        
        self.grid.grid_disp = []
        for x in range(self.grid.grid_size):
            for y in range(self.grid.grid_size):

                var= tk.StringVar()
                var.set(str(self.grid.map[x][y]))
                
                tk.Entry(root, text = var, bg=colour_map[self.grid.map[x][y]]
                ).grid(row=y+2, column = x+1)
        tk.mainloop()
        
    def display_grid(self):
        """
        trying to use tkinter to display gamme.. update script..
        """
        for n in range(len(self.grid_lst)):
            self.grid.grid_disp[n] = self.grid_lst[n]
            
    
    def display_grid_unix(self):
        """
        from CSMM 101X ColombiaX AI course, unix display code
        use grid array for this. 
        """
        grid_array = np.reshape(self.grid.lst, (self.grid.grid_size, self.grid.grid_size))
        for i in range(3 * self.grid.grid_size):
            for j in range(self.grid.grid_size):
                v = grid_array[int(i / 3)][j]
                if i % 3 == 1:
                    string = str(v).center(7, " ")
                else:
                    string = " "
                print(cTemp %  (colorMap[v], string), end="")
            print("")
            if i % 3 == 2:
                print("")
# a grid class, manages game logic, returns what moves are legal, 

class Grid(object):
    def __init__(self, grid_size=4):
        self.size = grid_size       #for minimax agent
        self.grid_size = grid_size
        self.lst = list(np.zeros(grid_size**2, dtype=int))
        self.map =  [[0] * self.size for i in range(self.size)] #for minimax agent
        self.moves = ['u', 'd', 'l', 'r']
    
    def clone(self):
        gridCopy = Grid()
        gridCopy.lst = copy.deepcopy(self.lst)
#        gridCopy.grid_size = self.grid_size
        return gridCopy    
    
    def __str__(self):
        grid_list = self.lst.copy()
        grid_str = 'x--x--x--x--x'
        for i in range(len(grid_list)):
            if i% self.grid_size == 0:
                grid_str = grid_str+'\n|'+ str(grid_list[i])+ ' '
            else:
                grid_str = grid_str+'|'+ str(grid_list[i])+' '
        grid_str = grid_str+'\nx--x--x--x--x'
        return grid_str
        
    def start(self):
        """
        places random 2 and 4s on the grid, effectively does two computer turns
        """
        t = 2
        for turn in range(t):
            self.comp_turn()

    def comp_turn(self):
        """
        places a random 2 or 1/10 chance of a 4 in any empty cell
        """
        cell_index = random.choice(self.find_empty_cells())
        if random.random()>0.9:
            self.lst[cell_index] = 4
        else:
            self.lst[cell_index] = 2
            
        
    def find_empty_cells(self):
        """
        more convenient to use list here
        returns list of cell indexes that are empty
        """
        empty_cells = []
        for i in range(len(self.lst)):
            if self.lst[i] == 0:
                empty_cells.append(i)
            else:
                continue
        return empty_cells
    
    def set_cell_val(self, tup, val):
        """
        to set a cell indicated by tup to a certain value
        """
        x,y = tup
        n = y*4 + x
        self.lst[n] = val
        self.map[x][y] = val

    def can_move(self):
        """
        returns boolean, true if you can move
        iterates through list pairs to look for pairs the same, 
        then rotates and tries again. 
        first checks for empty cells
        usually faster than running can_move_dir for each direction? i think..
        """
        if len(self.find_empty_cells()) != 0:
            return True
        else:
            grid_array = np.reshape(self.lst, (self.grid_size, self.grid_size))
            for lst in grid_array:
                for i, j in zip(lst, lst[1:]):
                    if i == j:
                        return True      
            rot_grid_array = np.rot90(grid_array)
            for lst in rot_grid_array:
                for i,j in zip(lst, lst[1:]):
                    if i == j:
                        return True       
        return False
    
    def get_available_moves(self):
        """
        tests each move for vaildity, returns list of valid moves
        """
        available_moves = []
        for move in [0, 1, 2, 3]:
            if self.can_move_dir(move):
                available_moves.append(move)
#        print (available_moves, self.can_move())
        return available_moves

    def can_move_dir(self, move):
        """
        tests a single move for vailidiy, does it actually change the map?
        """
        old_self = self.clone()
#        print(move)
        old_self.move(move)
        if (old_self.lst == self.lst):
            return False
        else:
            return True
    
    def move(self, move_i):
        """
        moves grid, returns new grid , boolean to say if there has been a move, 
        and new score.
        only need to write one direction, can rotate and transpose for others
        easier to use array for move
        """
        grid_array = np.reshape(self.lst, (self.grid_size, self.grid_size))
#        print(move_i)
        move = move_dict[move_i]
        if move == 'l':
            new_grid, score_inc = self.merge_l(grid_array)
        elif move == 'r':
            flip_grid = np.fliplr(grid_array)
            flip_merged_grid, score_inc = self.merge_l(flip_grid)
            new_grid = np.fliplr(flip_merged_grid)           
        elif move =='u':
            turn_grid = np.rot90(grid_array)
            turn_merge_grid, score_inc = self.merge_l(turn_grid)
            new_grid = np.rot90(turn_merge_grid, axes=(1,0))
        elif move =='d':
            turn_flip_grid = np.rot90(grid_array, axes=(1,0))
            t_f_merge_grid, score_inc = self.merge_l(turn_flip_grid)
            new_grid = np.rot90(t_f_merge_grid)
        
        self.lst = list(new_grid.flatten())
#        mapLst = []
#        for lst in new_grid:
#            mapLst.append(list(lst))
#        self.map = mapLst
        return score_inc
    
    def merge_l(self, grid_array):
        """
        takes in grid array, returns moved grid list, does not change self.map
        """
        score_inc = 0
        new_grid = []
        for r in grid_array:
            shifted_row=self.shift_row_left(r)
            merged_row, score = self.merge_row_left(shifted_row)
            score_inc +=score
            shifted_row=self.shift_row_left(merged_row)
            new_grid.append(shifted_row)
        g = np.array(new_grid)
        return g, score_inc
               
    def shift_row_left(self, row, c_start=0, c_end=1):
        """
        just shift one row left:
        """
#        print('start pos', c_start, 'val', row[c_start], \
#                  'end pos', c_end,'end val', row[c_end] )
        if row[c_start] !=0:
            try:
                return self.shift_row_left(row, c_start+1, c_start+2)
            except IndexError:
                return row
        if row[c_end] == 0:
            try:
                return self.shift_row_left(row, c_start, c_end+1)
            except IndexError:
                try:
                    return self.shift_row_left(row, c_start+1, c_start+2)
                except IndexError:
                    return row
        else:
            row[c_start] = row[c_end]
            row[c_end] = 0
            try:
                return self.shift_row_left(row, c_start+1, c_start+2)
            except IndexError:
                return row
                       
    def merge_row_left(self, row,c_start=0, score=0):
        """
        takes shifted grid array and merges left those tiles that are the same
        returns grid with spaces where there were numbers, this will be shifted afterwards.
        also returns score
        """
        try:
            if row[c_start]== row[c_start+1]:
                row[c_start] = 2*row[c_start]
                score += row[c_start]
                row[c_start+1] =0
                try:
                    return self.merge_row_left(row,c_start+2, score)
                except IndexError:
                    return row
            else:
                try:
                    return self.merge_row_left(row, c_start+1, score)
                except IndexError:
                    return row
        except IndexError:
            return row, score       
    
    def get_max_tile(self):
        return max(self.lst)
    
#    def getMaxTile(self):
#        """
#        so the minimax agent will work properly
#        """
#        max_tile = max(self.lst)
#        return max_tile
    
    def getAvailableCells(self):
        """
        again so the minimax agent will work properly
        """
        lst = self.find_empty_cells()
        new_lst = []
        for n in lst:
            i = n%4
            j = n//4
            new_lst.append((i,j))
        return new_lst
    
    def getAvailableMoves(self):
        """
        for minimax
        """
        return self.get_available_moves()
    
    def setCellValue(self, cell, val):
        """
        for minimax
        """
#        print('cell',cell)
        return  self.set_cell_val(cell, val)
    
    def canMove(self):
        """
        """
        return self.can_move()
            
        

class Agent(object):
    def get_move(self,grid):
        
        pass
    
class NNAgent(Agent):
    def get_move(self, grid):
        lamb = 0.1
        alpha = 0.03
        try:
            nnet = NN2048.NN.from_file(['w0.npy','w1.npy','w2.npy'])
        except:
            nnet = NN2048.NN()
#        print(type(nnet))
        move, output = nnet.get_move(grid)
        nnet.update_weights(grid,lamb,alpha)
        nnet.write_thetas() 
        return move

    
class MiniMaxAgent(Agent):
    def get_move(self, grid):
        agent = PlayerAI_3.PlayerAI()
        move = agent.getMove(grid)
        return move
    
    
class RandomAgent(Agent):
    def get_move(self, grid):
        move = random.choice(grid.get_available_moves())
        return move
    
class Human(Agent):
    def get_move(self, grid):
        move_key = None
        while not move_key:
            move = input('player chose move, u, d, l, r: (using w,s,a,d)')
            if move not in ['w','s','a','d']:
                print('not a valid move, please try again')
                continue
            else:
                move_key = move
#        human_move_dict = {'w' : 'u', 's':'d', 'a':'l', 'd':'r'}
        move = human_move_dict[move_key]
        return move
        
        
if __name__ == '__main__':
    results = []
    for a in ['NN']:
        lst = []
        for n in range(1):
            
            g = Game(agent = a)
            score, max_val = g.start()
            lst.append(max_val)
            print(max_val)
        results.append(lst)
            
    print(results)        
        
#    print(g.grid.map)
#    print(g.grid)
#    g.grid.move('l')
#    print(g.grid)
#    g.grid.move('r')
#    print(g.grid)
#    g.grid.move('u')
#    print(g.grid)
#    g.grid.move('d')
#    print(g.grid)
    