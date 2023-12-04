import numpy as np
from Simulation.SCurveInterpolator import SCurveInterpolator
from neura_dual_quaternions import DualQuaternion, Quaternion

class MoveJoint:
        def __init__(self, q_target, total_Time):
                self.q0 = np.zeros(7)
                self.q1 = q_target
                self.q0_set = False
                self.total_Time = total_Time
                self.done = False
                self.scurve = SCurveInterpolator()
                self.x0 = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.type = "joint"
        
        def evaluate(self, time):
                if time < self.total_Time:
                        self.q, self.q_dot, self.q_ddot = self.interpolateJoint(self.q0, self.q1, self.total_Time, time)
                else:
                        self.q = self.q1
                        self.q_dot = np.zeros(7)
                        self.q_ddot = np.zeros(7)
                        self.done = True
                
                return self.q, self.q_dot
    
        def interpolateJoint(self, q0, q1, total_Time, time):
                s, s_dot, s_ddot = self.scurve.evaluate(0, 1, total_Time, 0.4, 0.3, time)

                dq = q1 - q0
                q_interpolated = q0 + dq*s
                q_dot_interpolated = dq*s_dot
                q_ddot_interpolated = dq*s_ddot

                return q_interpolated, q_dot_interpolated, q_ddot_interpolated
    

        
    