import numpy as np
from Simulation.SCurveInterpolator import SCurveInterpolator
from neura_dual_quaternions import Quaternion, DualQuaternion


class LinearInterpolator:
        def __init__(self, x0, x1, total_Time):
                self.x0 = x0
                self.x1 = x1
                self.total_Time = total_Time
                self.scurve = SCurveInterpolator()
    
    
        def evaluateDQ(self, time):
                s, s_dot, s_ddot = self.scurve.evaluate(0, 1, self.total_Time, 0.4, 0.3, time)
                
                Qd = DualQuaternion.sclerp(self.x0, self.x1, s)
                Qd_dot = DualQuaternion.sclerp_dot(self.x0, self.x1, s, s_dot)
                
                # currently not implemented
                # Qd_ddot = DualQuaternion.sclerp_ddot(x0, x1, s, s_dot)
                Qd_ddot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                
                return Qd, Qd_dot, Qd_ddot
        
    