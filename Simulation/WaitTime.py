 import numpy as np
from neura_dual_quaternions import DualQuaternion

class WaitTime:
        
        # Initialize the WaitTime class with total waiting time
        def __init__(self, total_Time):
                self.total_Time = total_Time  # Total time to wait
                self.done = False  # Flag to indicate if the waiting is done
                self.set_x = False  # Flag to check if some position/state has been set (unused in this snippet)

                # Initializing the dual quaternions for initial state, desired state, and the rate of change of desired state
                self.x0 = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)  # Initial state (no rotation or translation)
                self.x_des = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)  # Desired state (no rotation or translation)
                self.x_des_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)  # Rate of change of desired state (zero)

                self.type = "wait"  # Type of operation, here it's a "wait"
    
        # Method to evaluate the current state based on the provided time
        def evaluate(self, time):
                # Check if the current time is less than the total time to wait
                if time < self.total_Time:
                        pass  # If waiting time is not yet over, do nothing
                else:
                        self.done = True  # If waiting time is over, mark as done

                # Return the initial state and the rate of change of the desired state
                return self.x0, self.x_des_dot
