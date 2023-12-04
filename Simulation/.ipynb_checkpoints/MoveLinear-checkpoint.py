import numpy as np
import time
from Simulation.LinearInterpolator import LinearInterpolator
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from Simulation.MPC_DifferentialKinematics import MPC_DifferentialKinematics
from neura_dual_quaternions import Quaternion, DualQuaternion


class MoveLinear:
        def __init__(self, x_target, total_Time):
                self.x0 = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.x1 = x_target
                
                self.q = np.zeros(7)
                self.forward_kinematics = ForwardKinematics()
                self.diffkin = DifferentialKinematics()
                
                self.total_Time = total_Time
                self.time = 0
                self.done = False

                self.Qd = None
                self.Qd_dot = None
                self.current_position = np.zeros(7)
                self.current_velocity = np.zeros(7)
                self.current_acceleration = np.zeros(7)
                self.current_cartesian_position = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)

        
        def run(self, dt):
                if self.time < self.total_Time:
                        self.time += dt
                        self.DQd, self.DQd_dot, self.DQd_ddot = self.trajectory.evaluateDQ(self.time)
                        
                        #self.current_velocity = self.diffkin.differential_kinematics(self.current_position, self.current_velocity, self.DQd, self.DQd_dot)
                        self.current_velocity = self.diffkin.quadratic_program_2(self.current_position, self.current_velocity, self.DQd, self.DQd_dot)
                        #self.current_velocity = self.diffkin.qrmc(self.current_position, self.current_velocity, self.DQd, self.DQd_dot, dt)
                        
                        self.current_position = self.current_position + self.current_velocity*dt
                        self.current_cartesian_position = self.DQd
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
                print("THIS IS TRIGGERED")
                self.current_position = position

        def setStartCartPosition(self, x):
                self.x0 = x
                self.trajectory = LinearInterpolator(self.x0, self.x1, self.total_Time)

        def getCartesianTarget(self):
                return self.current_cartesian_position
        
    