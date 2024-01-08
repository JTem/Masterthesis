import numpy as np
from Simulation.SCurveInterpolator import SCurveInterpolator
from neura_dual_quaternions import Quaternion, DualQuaternion


class LinearInterpolator:
        # Initialize the LinearInterpolator object with start and end positions, and total time for the movement
        def __init__(self, x0, x1, total_Time):
                self.x0 = x0 # Starting dual quaternion (position and orientation)
                self.x1 = x1 # Ending dual quaternion (position and orientation)
                self.total_Time = total_Time # Total time allocated for the interpolation
                self.scurve = SCurveInterpolator() # SCurveInterpolator for generating smooth interpolations
    
        # Evaluate the linear interpolation at a given time using dual quaternions
        def evaluateDQ(self, time):
                # Use the SCurveInterpolator to calculate scale factors for interpolation
                s, s_dot, s_ddot = self.scurve.evaluate(0, 1, self.total_Time, 0.4, 0.3, time)
                
                # Perform spherical linear interpolation (slerp) between two dual quaternions at scale factor s
                Qd = DualQuaternion.sclerp(self.x0, self.x1, s) # Interpolated dual quaternion
                Qd_dot = DualQuaternion.sclerp_dot(self.x0, self.x1, s, s_dot) # Derivative of the interpolated dual quaternion
                
                # Derivative of the dual quaternion is not implemented, so it's initialized as a basic constructor (zero)
                # This part is a placeholder for future implementations where second derivative might be calculated
                # Qd_ddot = DualQuaternion.sclerp_ddot(x0, x1, s, s_dot)
                Qd_ddot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                
                # Return the interpolated dual quaternion, its first and second derivatives
                return Qd, Qd_dot, Qd_ddot
        
    