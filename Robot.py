# -*- coding: utf-8 -*-
"""
Robot class, WPManager class, and Simulation class

(c) S. Bertrand
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Robot:
    
    def __init__(self, x0, y0, theta0, id=0):
        self.id = id        
        # state
        self.x = x0  # x position coordinate in meters
        self.y = y0  # y position coordinate in meters
        self.theta = theta0  # orientation angle in rad

        #input
        self.V = 0.0  # module of linear speed in m/s (>=0)
        self.omega = 0.0  # angular speed in rad/s
    
    
    # integrate robot motion over one sampling period (Euler discretization)
    def integrateMotion(self, Te):        
        self.x = self.x + Te * self.V * math.cos(self.theta)    
        self.y = self.y + Te * self.V * math.sin(self.theta)
        self.theta = self.theta + Te * self.omega

    # set V control input (in m/s)
    def setV(self, V):
        self.V = V        
        
    # set omega control imput (in rad/s)
    def setOmega(self, omega):
        self.omega = omega
    
        
    def __repr__(self):
        """Display in command line"""
        message = "Robot: id:{}\n".format(self.id)
        message += " x: {}(m), y: {}(m), theta: {}(deg)".format(self.x, self.y, self.theta*180.0/math.pi)        
        return message
    

    def __str__(self):
        """Display with print function"""
        message = "Robot: id:{}\n".format(self.id)
        message += " x: {}(m), y: {}(m), theta: {}(deg)".format(self.x, self.y, self.theta*180.0/math.pi)        
        return message

# *** end of class Robot ****************************************************



class WPManager:
# Assumption: WPList is a list of [ [x coord, y coord]  ]    
    
    def __init__(self, WPList, epsilonWP):
        self.WPList = list(WPList)
        self.epsilonWP = epsilonWP  # threshold for switching test to next WP
        self.currentWP = self.WPList.pop(0) # current WP
        self.xr = self.currentWP[0] # x coordinate of current WP
        self.yr = self.currentWP[1] # y coordinate of current WP
        
        
    def switchToNextWP(self):
        if not(self.isWPListEmpty()):
            self.currentWP = self.WPList.pop(0)        
            self.xr = self.currentWP[0]
            self.yr = self.currentWP[1]


    def isWPListEmpty(self):
        if (len(self.WPList)==0):
            return True
        else:
            return False

        
    def distanceToCurrentWP(self, x, y):
        return np.sqrt( math.pow(x-self.xr,2) + math.pow(y-self.yr,2))


# *** end of class WPManager **************************************************


class RobotSimulation:
    
    def __init__(self, robot, t0=0.0, tf=10.0, dt=0.01):
        self.t0 = t0 # init time of simulation (in sec)
        self.tf = tf # final time of simulation (in sec)
        self.dt = dt # sampling period for numerical integration (in sec)
        self.t = np.arange(t0, tf, dt) # vector of time stamps

        # to save robot state and input during simulation
        #self.x = np.zeros_like(self.t)
        self.x = np.empty(len(self.t))
        self.x[:] = np.nan
        #self.y = np.zeros_like(self.t)
        self.y = np.empty(len(self.t))
        self.y[:] = np.nan
        self.theta = np.zeros_like(self.t)
        
        #self.V = np.zeros_like(self.t)
        self.V = np.empty(len(self.t))
        self.V[:] = np.nan
       # self.omega = np.zeros_like(self.t)
        self.omega = np.empty(len(self.t))
        self.omega[:] = np.nan
        # to save reference position of current WP during simulation
        #self.xr = np.zeros_like(self.t)
        self.xr = np.empty(len(self.t))
        self.xr[:] = np.nan
       # self.yr = np.zeros_like(self.t)
        self.yr = np.empty(len(self.t))
        self.yr[:] = np.nan
        # to save intermediate and computed values in control computation
        #self.Vr = np.zeros_like(self.t)
        self.Vr = np.empty(len(self.t))
        self.Vr[:] = np.nan
        #self.thetar = np.zeros_like(self.t)
        self.thetar = np.empty(len(self.t))
        self.thetar[:] = np.nan
        #self.omegar = np.zeros_like(self.t)
        self.omegar = np.empty(len(self.t))
        self.omegar[:] = np.nan
        
        # potential
        #self.potential = np.zeros_like(self.t)
        self.potential = np.empty(len(self.t))
        self.potential[:] = np.nan
        
        
        # index of current stored data (from 0 to len(self.t)-1 )
        self.currentIndex = 0



    def addDataFromRobot(self, robot):
        self.x[self.currentIndex] = robot.x
        self.y[self.currentIndex] = robot.y
        self.theta[self.currentIndex] = robot.theta
        self.V[self.currentIndex] = robot.V
        self.omega[self.currentIndex] = robot.omega
        self.currentIndex = self.currentIndex + 1


    def addData(self, robot, WPManager, Vr, thetar, omegar, potential):
        self.x[self.currentIndex] = robot.x
        self.y[self.currentIndex] = robot.y
        self.theta[self.currentIndex] = robot.theta
        self.V[self.currentIndex] = robot.V
        self.omega[self.currentIndex] = robot.omega

        self.xr[self.currentIndex] = WPManager.xr
        self.yr[self.currentIndex] = WPManager.yr
        
        self.Vr[self.currentIndex] = Vr
        self.thetar[self.currentIndex] = thetar
        self.omegar[self.currentIndex] = omegar
        
        self.potential[self.currentIndex] = potential

        self.currentIndex = self.currentIndex + 1

        

    def plotXY(self, figNo = 1, xmin=-25, xmax=25, ymin=-25, ymax=25):
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
        ax.plot(self.x, self.y, color = 'r')
        ax.grid(True)
        ax.set_ylabel('y (m)')
        ax.set_xlabel('x (m)')
        return fig,ax


    def plotPotential(self, figNo = 1):
        fig2, graphTab2 = plt.subplots(3)
        graphTab2[0].plot(self.t, self.potential, color = 'r')
        graphTab2[0].grid(True)
        graphTab2[0].set_ylabel('potential')
        graphTab2[0].set_xlabel('t (s)') 
        graphTab2[1].plot(self.x, self.potential, color = 'r')
        graphTab2[1].grid(True)
        graphTab2[1].set_ylabel('potential')
        graphTab2[1].set_xlabel('x (m)')
        graphTab2[2].plot(self.y, self.potential, color = 'r')
        graphTab2[2].grid(True)
        graphTab2[2].set_ylabel('potential')
        graphTab2[2].set_xlabel('y (m)')
        
        
        
    def plotPotential3D(self, figNo = 1):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.potential)#, marker='o')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('potential')
        
        

    def plotXYTheta(self, figNo = 1):
        fig2, graphTab2 = plt.subplots(3)
        graphTab2[0].plot(self.t, self.x, color = 'r')
        graphTab2[0].plot(self.t, self.xr, color = 'b')
        graphTab2[0].grid(True)
        graphTab2[0].set_ylabel('x (m)')
        graphTab2[1].plot(self.t, self.y, color = 'r')
        graphTab2[1].plot(self.t, self.yr, color = 'b')        
        graphTab2[1].grid(True)
        graphTab2[1].set_ylabel('y (m)')
        graphTab2[2].plot(self.t, self.theta*180/math.pi, color = 'r')
        graphTab2[2].plot(self.t, self.thetar*180/math.pi, color = 'b')
        graphTab2[2].grid(True)
        graphTab2[2].set_ylabel('theta (deg)')
        graphTab2[2].set_xlabel('t (s)')


    def plotVOmega(self, figNo=1):
        fig22, graphTab22 = plt.subplots(2)
        graphTab22[0].plot(self.t, self.V, color = 'r')
        graphTab22[0].grid(True)
        graphTab22[0].set_ylabel('V (m/s)')
        graphTab22[1].plot(self.t, self.omega*180/math.pi, color = 'r')
        graphTab22[1].grid(True)
        graphTab22[1].set_ylabel('omega (deg/s)')
        graphTab22[1].set_xlabel('t (s)')

    '''
    def runAnimation(self, epsilonWP, figNo=1):
        # Animation
        fig = plt.figure(figNo)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5, 5), ylim=(-5, 5))
        ax.grid()
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        robotBody, = ax.plot([], [], 'o-', lw=2)
        robotDirection, = ax.plot([], [], '-', lw=1, color='k')
        wayPoint, = ax.plot([], [], 'o-', lw=2, color='b')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        WPArea, = ax.plot([], [], ':', lw=1, color='b')

        thetaWPArea = np.arange(0.0,2.0*math.pi+2*math.pi/30.0, 2.0*math.pi/30.0)
        xWPArea = epsilonWP*np.cos(thetaWPArea)
        yWPArea = epsilonWP*np.sin(thetaWPArea)

        def initAnimation():
            print("init")
            print(len(self.t))
            robotDirection.set_data([], [])
            robotBody.set_data([], [])
            wayPoint.set_data([], [])
            WPArea.set_data([], [])
            robotBody.set_color('r')
            robotBody.set_markersize(20)    
            time_text.set_text('')
            return robotBody,robotDirection, wayPoint, time_text, WPArea  
            
        def animateFun(i):  
            print("animate")
            robotBody.set_data(self.x[i], self.y[i])          
            wayPoint.set_data(self.xr[i], self.yr[i])
            WPArea.set_data(self.xr[i]+xWPArea.transpose(), self.yr[i]+yWPArea.transpose())    
            thisx = [self.x[i], self.x[i] + 0.5*math.cos(self.theta[i])]
            thisy = [self.y[i], self.y[i] + 0.5*math.sin(self.theta[i])]
            robotDirection.set_data(thisx, thisy)
            time_text.set_text(time_template%(i*self.dt))
            return robotBody,robotDirection, wayPoint, time_text, WPArea

        ani = animation.FuncAnimation(fig, animateFun, np.arange(1, len(self.t)),
            interval=4, blit=True, init_func=initAnimation, repeat=False)
        '''

        
if __name__=='__main__':
    robot = Robot(0.0, 0.0, 0.0)
    print(robot)        
    robot.integrateMotion(0.05)
    print(robot)
    