import numpy as np
import time
from Simulation.LinearInterpolator import LinearInterpolator
from neura_dual_quaternions import Quaternion, DualQuaternion


class MoveLinear:
        def __init__(self, x_start, x_end, total_Time):
                self.x0 = x_start
                self.x1 = x_end
                self.total_Time = total_Time
                        
                self.trajectory = LinearInterpolator(self.x0, self.x1, self.total_Time)
          
                self.done = False
                
                self.type = "cart"
                
        
        def evaluate(self, time):
                if time < self.total_Time:
                        
                        self.x_des, self.x_des_dot, self.x_des_ddot = self.trajectory.evaluateDQ(time)
                else:
                        self.x_des, self.x_des_dot, self.x_des_ddot = self.trajectory.evaluateDQ(self.total_Time)
                        self.done = True
                
                return self.x_des, self.x_des_dot
            