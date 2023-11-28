import numpy as np
import cvxpy as cp
import time
from Interpolators import Interpolators
from ForwardKinematics import ForwardKinematics
from neura_dual_quaternions import Quaternion, DualQuaternion



class MoveTrajectory:
        def __init__(self, T_target, total_Time):
                self.T0 = np.eye(4)
                self.T1 = T_target

                self.q = np.zeros(7)
                self.forward_kinematics = ForwardKinematics()

                self.total_Time = total_Time
                self.time = 0
                self.done = False


                self.Qd = None
                self.Qd_dot = None
                self.current_position = np.zeros(7)
                self.current_velocity = np.zeros(7)
                self.current_acceleration = np.zeros(7)
                self.current_cartesian_postion = np.eye(4)
                self.interpolator = Interpolators()
        
        def run(self, dt):
                if self.time < self.total_Time:
                        self.time += dt
                        self.Qd, self.Qd_dot = self.interpolateDualQuaternion(self.T0, self.T1, self.total_Time, self.time)

                        self.current_velocity = self.differential_kinematics(self.current_position, self.current_velocity, self.Qd, self.Qd_dot)       

                        self.current_position = self.current_position + self.current_velocity*dt
                        self.current_cartesian_postion = self.Qd
                else:

                        self.current_velocity = np.zeros(7)
                        self.current_acceleration = np.zeros(7)
                        self.done = True

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
                self.current_position = position

        def setStartCartPosition(self, T):
                self.T0 = T

        def getCartesianTarget(self):
                return self.current_cartesian_postion
        
    