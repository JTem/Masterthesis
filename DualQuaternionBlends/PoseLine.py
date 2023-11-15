import math
import numpy as np
from DualQuaternions.Quaternion import Quaternion

class Line:
    def __init__(self, p1, p2, q1, q2, velocity, angular_velocity_max):
        self.p1 = p1
        self.p2 = p2
        self.q1 = q1
        self.q2 = q2
        self.velocity = velocity
        self.angular_velocity_max = angular_velocity_max
        
        self.dist =np.linalg.norm(self.p2-self.p1)
        self.tangent = np.array([0,0,0])
        if self.dist > 1e-6:
            self.tangent = (self.p2-self.p1)/self.dist
        
        
        self.duration = self.dist/self.velocity
        
        dq = q1.inverse()*q2
        self.theta = abs(dq.getAngle())
        e = dq.log()
        if e.norm() > 1e-6:
            e = e.normalize()
            
        e0 = q1*e*q1.inverse()
        self.rotation_axis = e0.getVector().flatten()
        
        if self.duration > 1e-6:
            self.angular_velocity = self.theta/self.duration
            
            if self.angular_velocity > self.angular_velocity_max:
                self.duration = self.theta / self.angular_velocity_max
                self.angular_velocity = self.angular_velocity_max
                
        else:
            self.angular_velocity = self.angular_velocity_max
            self.duration = self.theta / self.angular_velocity_max
        
        

            
        
        
        
        
    def getPos(self, time):
       # point_on_arc = np.array([self.radius*np.cos(time/self.duration*self.angle), self.radius*np.sin(time/self.duration*self.angle), 0])
        
        translation = self.p1 + self.getVel(0)*time
        
        return translation    
    
    def getAngularVel(self):
        return self.rotation_axis*self.theta/self.duration
    
    def getVel(self, time):
        return self.tangent*self.dist /self.duration
    
    def getAcc(self, time):
        return np.array([0,0,0])