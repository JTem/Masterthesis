import numpy as np
import time
from Simulation.LinearInterpolator import LinearInterpolator
from neura_dual_quaternions import Quaternion, DualQuaternion


class MoveLinear:
        
        def __init__(self, x_start, x_end, total_Time):
                self.x0 = x_start # Starting position of the movement
                self.x1 = x_end # Ending position of the movement
                self.total_Time = total_Time # Total time allocated for the movement
                
                # Create a LinearInterpolator object for generating linear motion between x0 and x1 over total_Time
                self.trajectory = LinearInterpolator(self.x0, self.x1, self.total_Time)
          
                self.done = False
                
                self.type = "cart"
                
        # Evaluate the linear trajectory at a given time
        def evaluate(self, time):
                # Check if the given time is within the trajectory duration
                if time < self.total_Time:
                        # Evaluate the trajectory for the given time to get desired position (x_des), velocity (x_des_dot), and acceleration (x_des_ddot)
                        self.x_des, self.x_des_dot, self.x_des_ddot = self.trajectory.evaluateDQ(time)
                else:
                        # If the given time exceeds the total trajectory time, evaluate at the last time point
                        self.x_des, self.x_des_dot, self.x_des_ddot = self.trajectory.evaluateDQ(self.total_Time)
                        self.done = True
                        
                # Return the desired position and velocity
                return self.x_des, self.x_des_dot
            