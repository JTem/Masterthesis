import math
import numpy as np
from DualQuaternionBlends.PoseLine import Line

class PoseL:
    def __init__(self, P, Q, velocity, angular_velocity_max):
        self.P = P
        self.velocity = velocity
        self.angular_velocity_max = angular_velocity_max
        self.line_list = []
        
        for i in range(0, len(P)-1):
            
           self.line_list.append(Line(P[i], P[i+1], Q[i], Q[i+1], self.velocity, self.angular_velocity_max))
           
           
    def getLineList(self):
        return self.line_list
    
