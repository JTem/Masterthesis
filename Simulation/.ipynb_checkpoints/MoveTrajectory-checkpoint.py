import numpy as np
from DualQuaternionQuinticBlends import DQQBTrajectoryGenerator
from neura_dual_quaternions import Quaternion, DualQuaternion



class MoveTrajectory:
        # Initialize the MoveTrajectory object with a given trajectory
        def __init__(self, trajectory):
                
                self.trajectory = trajectory # The trajectory to follow, a DQQBTrajectoryGenerator Object
                self.total_Time = self.trajectory.time_vector[-1] # The total time of the trajectory
                self.done = False
                
                self.type = "cart"

        # Evaluate the trajectory at a given time
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
    