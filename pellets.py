import pygame
from vector import Vector2
from constants import *
import numpy as np
import random
import math
class Pellet(object):
    def __init__(self, row, column, node_row, node_col):
        self.name = PELLET
        self.position = Vector2(column*TILEWIDTH, row*TILEHEIGHT)
        self.coord = (node_col, node_row)
        self.color = WHITE
        self.row_col = (column, row)
        
        self.radius = int(2 * TILEWIDTH / 16)
        self.collideRadius = int(2 * TILEWIDTH / 16)
        self.points = 10
        self.visible = True
        
    def render(self, screen):
        if self.visible:
            adjust = Vector2(TILEWIDTH, TILEHEIGHT) / 2
            p = self.position + adjust
            pygame.draw.circle(screen, self.color, p.asInt(), self.radius)


class PowerPellet(Pellet):
    def __init__(self, row, column):
        Pellet.__init__(self, row, column, row, column)
        self.name = POWERPELLET
        self.radius = int(8 * TILEWIDTH / 16)
        self.points = 50
        self.flashTime = 0.2
        self.timer= 0
        
    def update(self, dt):
        self.timer += dt
        if self.timer >= self.flashTime:
            self.visible = not self.visible
            self.timer = 0



class PelletGroup(object):
    def __init__(self, pelletfile, nodes):
        self.pelletList = []
        self.powerpellets = []
        self.nodegroup = nodes
        self.createPelletList(pelletfile)
        self.numEaten = 0
        

    def update(self, dt):
        for powerpellet in self.powerpellets:
            powerpellet.update(dt)
                
    def createPelletList(self, pelletfile):
        data = self.readPelletfile(pelletfile)
              
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row][col] in ['+']:
                    """
                    

                    best_node = None
                    best_dist = 100000
                    for node in self.nodegroup.nodesLUT.values():
                        dist = math.sqrt((row - node.coords[0])**2 + (col - node.coords[1])**2)
                        if dist < best_dist:
                            best_node = node
                            best_dist = dist
                    
                    node_row = best_node.coords[0]
                    node_col = best_node.coords[1]
                    good = True
                    for pell in self.pelletList:
                        if pell.coord == (node_col, node_row):
                            good = True
                            
                    if good:
                        self.pelletList.append(Pellet(row, col, node_row, node_col))
                    """
                elif data[row][col] in ['P', 'p']:
                    pp = PowerPellet(row, col)
                    self.pelletList.append(pp)
                    self.powerpellets.append(pp)
        count = 0  
        while(count < 20): #controls num of pellets added
            #print("Count = ", count)
            row = round(random.random()*data.shape[0]) - 1
            col = round(random.random()*data.shape[1]) - 1
            if data[row][col] in ['+']:
                #for pell in self.pelletList:
                  #  if pell.coord != (col,row):
                best_node = None
                best_dist = 100000
                for node in self.nodegroup.nodesLUT.values():
                    dist = math.sqrt((row - node.coords[0])**2 + (col - node.coords[1])**2)
                    if dist < best_dist:
                        best_node = node
                        best_dist = dist
                  
                node_row = best_node.coords[0]
                node_col = best_node.coords[1]
                good = True
                for pell in self.pelletList:
                    if pell.coord == (node_col, node_row):
                        good = True
                        
                if good:
                    self.pelletList.append(Pellet(row, col, node_row, node_col))
                    count = count + 1
        #for pell in self.pelletList:
            #print("Coord = ", pell.coord, "Col_Row = ", pell.row_col)
    def readPelletfile(self, textfile):
        return np.loadtxt(textfile, dtype='<U1')
    
    def isEmpty(self):
        if len(self.pelletList) == 0:
            return True
        return False
    
    def render(self, screen):
        for pellet in self.pelletList:
            pellet.render(screen)