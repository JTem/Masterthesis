import numpy as np
from DualQuaternionQuinticBlends import DQQBTrajectoryGenerator
from neura_dual_quaternions import Quaternion, DualQuaternion



class MoveTrajectory:
        def __init__(self, trajectory):
                
                self.trajectory = trajectory        
                self.total_Time = self.trajectory.time_vector[-1]
                self.done = False
                
                self.type = "cart"

        
        def evaluate(self, time):
                if time < self.total_Time:
                        self.x_des, self.x_des_dot, self.x_des_ddot = self.trajectory.evaluateDQ(time)

                else:
                        self.x_des, self.x_des_dot, self.x_des_ddot = self.trajectory.evaluateDQ(self.total_Time)
                        self.done = True
                
                return self.x_des, self.x_des_dot
    