# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:10:16 2021

(c) S. Bertrand
"""


import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal



class Potential:

    def __init__(self, difficulty=1, random=False):
        
        if (difficulty<1)or(difficulty>3):
            raise NameError("Difficulty must be >=1 and <=3")
        
        self.difficulty = difficulty
        self.random = random
        
        self.xmin = -25.
        self.xmax = 25.
        self.xstep = 0.05
        self.ymin = -25.
        self.ymax = 25.
        self.ystep = 0.05
        
        
        
        if (random):
            xwidth = np.abs(self.xmax - self.xmin)
            ywidth = np.abs(self.ymax - self.ymin)
            self.mu1 = [ 0.6*(xwidth*np.random.rand()-xwidth/2.) , 0.6*(ywidth*np.random.rand()-ywidth/2.) ]
            self.mu2 = [ xwidth*np.random.rand()-xwidth/2. , ywidth*np.random.rand()-ywidth/2. ]
            self.mu3 = [ xwidth*np.random.rand()-xwidth/2. , ywidth*np.random.rand()-ywidth/2. ]
        else:
            self.mu1 = [6, 4]
            self.mu2 = [-2, -2]
            self.mu3 = [-7, 10]
        
        self.gaussian1 = multivariate_normal(self.mu1, [[1.0, 0.], [0., 1.]])
        self.gaussian2 = multivariate_normal(self.mu2, [[0.5, 0.3], [0.3, 0.5]])
        self.gaussian3 = multivariate_normal(self.mu3, [[0.8, 0.], [0., 0.8]])
        
        self.weight1 = 10000
        self.weight2 = 1
        self.weight3 = 1E-8
        
        
        self.mu = [self.mu1]
        if (difficulty>1):
            self.mu.append(self.mu2)
            if (difficulty>2):
                self.mu.append(self.mu3)
        
        self.distribution = [self.gaussian1]
        if (difficulty>1):
            self.distribution.append(self.gaussian2)
            if (difficulty>2):
                self.distribution.append(self.gaussian3)
                
        self.weight = [self.weight1]
        if (difficulty>1):
            self.weight.append(self.weight2)
            if (difficulty>2):
                self.weight.append(self.weight3)


    
    
    def value(self,pos):  # (pos = [x,y]) 
        
        sumval = 0.
        
        for i in range(self.difficulty):
            sumval += self.weight[i]*self.distribution[i].pdf(pos)
        
        return np.fmax(310.+np.log10(sumval), -10.)



    def plot(self,noFigure=None,fig=None,ax=None):

        x, y = np.mgrid[self.xmin:self.xmax:self.xstep, self.ymin:self.ymax:self.ystep]
        pos = np.dstack((x, y))
        potentialFieldForPlot = self.value(pos)
        
        if (fig==None):
            if (noFigure==None):
                noFigure=1
            fig = plt.figure(noFigure)
        if (ax==None):
            ax = fig.add_subplot(111)
        cs = ax.contourf(x, y, potentialFieldForPlot, 20, cmap='BrBG')
        
        fig.colorbar(cs)
        
        return fig, ax






# main
if __name__=='__main__':

    
    plt.close()
    
    pot = Potential(difficulty=2, random=False)
       
    fig2, ax2 = pot.plot(1)
    

    print(pot.value( [0. , 0.] ))
    print(pot.value( [-2., 2.]) )

    
    plt.show()
