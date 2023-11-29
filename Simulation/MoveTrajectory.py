import numpy as np
from Simulation.Interpolators import Interpolators
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from DualQuaternionQuinticBlends import DQQBTrajectoryGenerator
from neura_dual_quaternions import Quaternion, DualQuaternion



class MoveTrajectory:
        def __init__(self, trajectory):
                
                self.trajectory = trajectory
                self.interpolator = Interpolators()
                self.diffkin = DifferentialKinematics()
                
                self.q = np.zeros(7)
                self.forward_kinematics = ForwardKinematics()
                Tj, Ta, Tv, T = self.interpolator.calculateTotalTime(0, self.trajectory.time_vector[-1], 1, 20, 1000)
                self.alpha = Ta/T
                self.beta = Tj/Ta
        
                self.total_Time = T
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
                        
                        s, _, _ = self.interpolator.timeScaling_S_single(0, self.trajectory.time_vector[-1], self.total_Time, self.alpha, self.beta, self.time)
                        self.DQd, self.DQd_dot, self.Qd_ddot = self.trajectory.evaluateDQ(s)

                        self.current_velocity = self.diffkin.differential_kinematics(self.current_position, self.current_velocity, self.DQd, self.DQd_dot)       

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
                self.current_position = position

        def setStartCartPosition(self, T):
                self.T0 = T

        def getCartesianTarget(self):
                return self.current_cartesian_position
        
    