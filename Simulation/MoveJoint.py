import numpy as np
from Simulation.Interpolators import Interpolators
from neura_dual_quaternions import DualQuaternion, Quaternion

class MoveJoint:
        def __init__(self, q_target, total_Time):
                self.q0 = np.zeros(7)
                self.q1 = q_target
                self.total_Time = total_Time
                self.time = 0
                self.done = False
                self.current_position = np.zeros(7)
                self.current_velocity = np.zeros(7)
                self.current_acceleration = np.zeros(7)
                self.current_cartesian_position = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.interpolator = Interpolators()
        
        def run(self, dt):
                if self.time < self.total_Time:
                        self.time += dt
                        self.current_position, self.current_velocity, self.current_acceleration = self.interpolateJoint(self.q0, self.q1, self.total_Time, self.time)
                else:
                        self.current_position = self.q1
                        self.current_velocity = np.zeros(7)
                        self.current_acceleration = np.zeros(7)
                        self.done = True
    
        def interpolateJoint(self, q0, q1, total_Time, time):
                s, s_dot, s_ddot = self.interpolator.timeScaling_S_single(0, 1, total_Time, 0.4, 0.3, time)

                dq = q1 - q0
                q_interpolated = q0 + dq*s
                q_dot_interpolated = dq*s_dot
                q_ddot_interpolated = dq*s_ddot

                return q_interpolated, q_dot_interpolated, q_ddot_interpolated
    
        def reset(self):
                self.time = 0
                self.done = False

        def getPosition(self):
                return self.current_position

        def getVelocity(self):
                return self.current_velocity

        def getAcceleration(self):
                return self.current_acceleration

        def setStartPosition(self, position):
                self.q0 = position
                self.current_position = position

        def setStartCartPosition(self, x):
                self.current_cartesian_position = x

        def getCartesianTarget(self):
                return self.current_cartesian_position
        
    