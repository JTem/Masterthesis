import numpy as np
from Simulation.SCurveInterpolator import SCurveInterpolator
from neura_dual_quaternions import DualQuaternion, Quaternion

class MoveJoint:
        
        # Initialize the MoveJoint object with target joint positions and total time for the movement
        def __init__(self, q_target, total_Time):
                
                self.q0 = np.zeros(7) # Initialize the starting joint positions as zeros
                self.q1 = q_target # Target joint positions
                self.q0_set = False # Flag to indicate if the initial joint positions have been set
                self.total_Time = total_Time # Total time allocated for the joint movement
                self.done = False
                self.scurve = SCurveInterpolator() # SCurveInterpolator for generating smooth joint movements
                self.x0 = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0) # Initial transformation (no translation or rotation)
                self.type = "joint"
        
        
        # Evaluate the joint trajectory at a given time
        def evaluate(self, time):
                
                # Check if the given time is within the trajectory duration
                if time < self.total_Time:
                        
                        # Interpolate the joint positions, velocities, and accelerations at the given time
                        self.q, self.q_dot, self.q_ddot = self.interpolateJoint(self.q0, self.q1, self.total_Time, time)
                else:
                        
                        # If the given time exceeds the total trajectory time, set to target positions and zero velocities
                        self.q = self.q1
                        self.q_dot = np.zeros(7)
                        self.q_ddot = np.zeros(7)
                        self.done = True
                        
                # Return the interpolated joint positions and velocities
                return self.q, self.q_dot
    
        def interpolateJoint(self, q0, q1, total_Time, time):
                s, s_dot, s_ddot = self.scurve.evaluate(0, 1, total_Time, 0.4, 0.3, time)

                dq = q1 - q0
                q_interpolated = q0 + dq*s
                q_dot_interpolated = dq*s_dot
                q_ddot_interpolated = dq*s_ddot

                return q_interpolated, q_dot_interpolated, q_ddot_interpolated
    

        
    