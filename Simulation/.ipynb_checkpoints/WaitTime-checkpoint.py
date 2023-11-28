import numpy as np
from neura_dual_quaternions import DualQuaternion

class WaitTime:
        def __init__(self, wait_time):
                self.wait_time = wait_time
                self.time = 0
                self.done = False
                self.running = False
                self.current_position = np.zeros(7)
                self.current_velocity = np.zeros(7)
                self.current_acceleration = np.zeros(7)
                self.current_cartesian_position = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
        
        def run(self, dt):
                if self.time < self.wait_time:
                        self.time += dt
                else:
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

        def setStartCartPosition(self, x):
                  self.current_cartesian_position = x

        def getCartesianTarget(self):
                return self.current_cartesian_position