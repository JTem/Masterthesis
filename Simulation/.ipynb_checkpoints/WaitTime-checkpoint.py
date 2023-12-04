import numpy as np
from neura_dual_quaternions import DualQuaternion

class WaitTime:
        
        def __init__(self, total_Time):
                self.total_Time = total_Time
                self.done = False
                self.set_x = False;
                
                self.x0 = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                
                self.x_des = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.x_des_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                
                self.type = "wait"
        
        def evaluate(self, time):
                
                if time < self.total_Time:
                        pass
                else:
                        self.done = True
                        
                return self.x0, self.x_des_dot
                
    
