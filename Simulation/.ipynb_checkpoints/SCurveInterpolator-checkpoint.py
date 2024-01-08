import math
import numpy as np

class SCurveInterpolator:
        # Initialize the SCurveInterpolator object
        def __init__(self):
                self.name = "interpolator"
        
         # Calculate the total time needed for the interpolation based on given parameters
        def calculateTotalTime(self, q0_, q1_, v_max, a_max, j_max):
                # Initialization of time segments for the s-curve profile
                Tj = 0 # Time of jerk phase
                Ta = 0 # Time of acceleration phase
                Tv = 0 # Time of constant velocity phase
                T = 0 # Total time
                
                # Calculate the direction of movement
                sigma = sigma = math.copysign(1, q1_ - q0_)
                
                # Normalize start and end positions
                q0 = sigma*q0_
                q1 = sigma*q1_
                
                # Compute time durations based on maximum velocity, acceleration, and jerk
                # There are different cases depending on the relationship between these parameters
                if v_max*j_max >= a_max**2:
                        Tj = a_max/j_max
                        Ta = Tj + v_max/a_max
                else:
                        Tj = np.sqrt(v_max/j_max)
                        Ta = 2.0*Tj
                
                # Calculate the time for constant velocity phase
                Tv = (q1 - q0)/v_max - Ta
                
                # Adjust the times if the constant velocity phase is not needed
                if Tv < 0:
                        Tv = 0
                        # Different cases for the distance to be covered and maximum acceleration and jerk
                        if (q1 - q0) >= 2.0*(a_max**3/j_max**2):
                                Tj = a_max/j_max
                                Ta = 0.5*Tj + np.sqrt((0.5*Tj)**2 + (q1 - q0)/a_max)
                        else:
                                Tj = ((q1 - q0)/(2.0*j_max))**(1.0/3.0)
                                Ta = 2.0*Tj
                # Calculate total time by summing all phases
                T = Tv + 2.0*Ta
                
                # Return all time parameters
                return Tj, Ta, Tv, T
        
        
        # Evaluate the interpolation at a given time t
        def evaluate(self, q0_, q1_, T, alpha, beta, t):
                # Calculate the direction of the motion
                sigma = math.copysign(1, q1_ - q0_)
                
                # If there is no movement, return the start position and zero velocities
                if sigma == 0.0:
                        res = [q0_, 0.0, 0.0, 0.0]
                        return res
                
                # Normalize start and end positions
                q0 = sigma * q0_
                q1 = sigma * q1_
                
                # Calculate the total distance to move
                h = q1 - q0
                
                # Calculate maximum jerk needed based on the interpolation parameters
                j_max = h / (alpha * alpha * beta * (1.0 - alpha) * (1.0 - beta) * T * T * T)
                j_min = -j_max
                
                # Calculate times for acceleration phase and jerk phase
                Ta = alpha * T
                Tj = beta * Ta
                
                # Calculate the time for the constant velocity phase
                Tv = T - 2 * Ta

                if h > 0.0:
                        Tv = T - 2.0 * Ta
                
                # Calculate the maximum acceleration and velocity limits based on jerk and time
                alim = j_max * Tj
                vlim = (Ta - Tj) * alim
                
                # Initialize other variables for deceleration and jerk phases
                Td = Ta
                Tj1 = Tj
                Tj2 = Tj
                
                # Initial velocity and final velocity for the profile
                v0 = 0.0
                v1 = 0.0
                alim_a = alim # Maximum acceleration during acceleration phase
                alim_d = -alim # Maximum acceleration during deceleration phase
                
                # Initialize positions and velocities
                q = 0
                q_dot = 0
                q_ddot = 0

                # Constrain time within 0 to T
                if t < 0:
                        t = 0
                if t > T:
                        t = T
                
                # Calculate position, velocity, and acceleration based on the current time
                # The following are different segments of the s-curve profile
                # Each segment defines the position, velocity, and acceleration based on the equations of motion for that segment
                # The segments include initial jerk, acceleration, constant velocity, deceleration, and final jerk phases
                # Each if statement below corresponds to one of these phases
                
                # Acceleration phase with increasing jerk
                if 0.0 <= t <= Tj1:
                        q = q0 + v0 * t + j_max * (t ** 3 / 6)
                        q_dot = v0 + 0.5 * j_max * t ** 2
                        q_ddot = j_max * t
                        #q_dddot = j_max
                
                # Constant acceleration phase
                if Tj1 <= t <= Ta - Tj1:
                        q = q0 + v0 * t + alim_a / 6 * (3 * t ** 2 - 3 * Tj1 * t + Tj1 ** 2)
                        q_dot = v0 + alim_a * (t - 0.5 * Tj1)
                        q_ddot = alim_a
                        # q_dddot = 0
                
                # Acceleration phase with decreasing jerk
                if Ta - Tj1 <= t <= Ta:
                        q = q0 + (vlim + v0) * 0.5 * Ta - vlim * (Ta - t) - j_min * ((Ta - t) ** 3 / 6)
                        q_dot = vlim + j_min * 0.5 * (Ta - t) ** 2
                        q_ddot = -j_min * (Ta - t)
                        #q_dddot = j_min
                
                # Constant velocity phase
                if Ta <= t <= Ta + Tv:
                        q = q0 + (vlim + v0) * 0.5 * Ta + vlim * (t - Ta)
                        q_dot = vlim
                        q_ddot = 0
                        #q_dddot = 0
                
                # Deceleration phase with increasing negative jerk
                if T - Td <= t <= T - Td + Tj2:
                        q = q1 - (vlim + v1) * 0.5 * Td + vlim * (t - T + Td) - j_max * ((t - T + Td) ** 3) / 6
                        q_dot = vlim - j_max * 0.5 * (t - T + Td) ** 2
                        q_ddot = -j_max * (t - T + Td)
                        # q_dddot = j_min
                
                # Constant deceleration phase
                if T - Td + Tj2 <= t <= T - Tj2:
                        q = q1 - (vlim + v1) * 0.5 * Td + vlim * (t - T + Td) + (alim_d / 6) * (
                        3 * (t - T + Td) ** 2 - 3 * Tj2 * (t - T + Td) + Tj2 ** 2
                        )
                        q_dot = vlim + alim_d * (t - T + Td - Tj2 * 0.5)
                        q_ddot = alim_d
                        #q_dddot = 0
                
                # Deceleration phase with decreasing negative jerk
                if T - Tj2 <= t <= T:
                        q = q1 - v1 * (T - t) - j_max * ((T - t) ** 3) / 6
                        q_dot = v1 + j_max * 0.5 * (T - t) ** 2
                        q_ddot = -j_max * (T - t)
                        #q_dddot = j_max

                # Return the position, velocity, and acceleration all scaled back to original direction
                return sigma * q, sigma * q_dot, sigma * q_ddot