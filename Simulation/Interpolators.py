import math
import numpy as np

class Interpolators:
        def __init__(self):
                self.name = "interpolator"
        
        
        def calculateTotalTime(self, q0_, q1_, v_max, a_max, j_max):
                Tj = 0
                Ta = 0
                Tv = 0
                T = 0
                
                sigma = sigma = math.copysign(1, q1_ - q0_)
                
                q0 = sigma*q0_
                q1 = sigma*q1_
                
                if v_max*j_max >= a_max**2:
                        Tj = a_max/j_max
                        Ta = Tj + v_max/a_max
                else:
                        Tj = np.sqrt(v_max/j_max)
                        Ta = 2.0*Tj
                
                Tv = (q1 - q0)/v_max - Ta
                
                if Tv < 0:
                        Tv = 0
                        if (q1 - q0) >= 2.0*(a_max**3/j_max**2):
                                Tj = a_max/j_max
                                Ta = 0.5*Tj + np.sqrt((0.5*Tj)**2 + (q1 - q0)/a_max)
                        else:
                                Tj = ((q1 - q0)/(2.0*j_max))**(1.0/3.0)
                                Ta = 2.0*Tj
                
                T = Tv + 2.0*Ta
                
                return Tj, Ta, Tv, T
        
        
        def timeScaling_S_single(self, q0_, q1_, T, alpha, beta, t):
                sigma = math.copysign(1, q1_ - q0_)

                if sigma == 0.0:
                        res = [q0_, 0.0, 0.0, 0.0]
                        return res

                q0 = sigma * q0_
                q1 = sigma * q1_

                h = q1 - q0

                j_max = h / (alpha * alpha * beta * (1.0 - alpha) * (1.0 - beta) * T * T * T)
                j_min = -j_max

                Ta = alpha * T
                Tj = beta * Ta
                Tv = T - 2 * Ta

                if h > 0.0:
                        Tv = T - 2.0 * Ta

                alim = j_max * Tj
                vlim = (Ta - Tj) * alim

                Td = Ta
                Tj1 = Tj
                Tj2 = Tj
                v0 = 0.0
                v1 = 0.0
                alim_a = alim
                alim_d = -alim

                q = 0
                q_dot = 0
                q_ddot = 0
                #q_dddot = 0

                if t < 0:
                        t = 0
                if t > T:
                        t = T

                if 0.0 <= t <= Tj1:
                        q = q0 + v0 * t + j_max * (t ** 3 / 6)
                        q_dot = v0 + 0.5 * j_max * t ** 2
                        q_ddot = j_max * t
                        #q_dddot = j_max

                if Tj1 <= t <= Ta - Tj1:
                        q = q0 + v0 * t + alim_a / 6 * (3 * t ** 2 - 3 * Tj1 * t + Tj1 ** 2)
                        q_dot = v0 + alim_a * (t - 0.5 * Tj1)
                        q_ddot = alim_a
                        # q_dddot = 0

                if Ta - Tj1 <= t <= Ta:
                        q = q0 + (vlim + v0) * 0.5 * Ta - vlim * (Ta - t) - j_min * ((Ta - t) ** 3 / 6)
                        q_dot = vlim + j_min * 0.5 * (Ta - t) ** 2
                        q_ddot = -j_min * (Ta - t)
                        #q_dddot = j_min

                if Ta <= t <= Ta + Tv:
                        q = q0 + (vlim + v0) * 0.5 * Ta + vlim * (t - Ta)
                        q_dot = vlim
                        q_ddot = 0
                        #q_dddot = 0

                if T - Td <= t <= T - Td + Tj2:
                        q = q1 - (vlim + v1) * 0.5 * Td + vlim * (t - T + Td) - j_max * ((t - T + Td) ** 3) / 6
                        q_dot = vlim - j_max * 0.5 * (t - T + Td) ** 2
                        q_ddot = -j_max * (t - T + Td)
                        # q_dddot = j_min

                if T - Td + Tj2 <= t <= T - Tj2:
                        q = q1 - (vlim + v1) * 0.5 * Td + vlim * (t - T + Td) + (alim_d / 6) * (
                        3 * (t - T + Td) ** 2 - 3 * Tj2 * (t - T + Td) + Tj2 ** 2
                        )
                        q_dot = vlim + alim_d * (t - T + Td - Tj2 * 0.5)
                        q_ddot = alim_d
                        #q_dddot = 0

                if T - Tj2 <= t <= T:
                        q = q1 - v1 * (T - t) - j_max * ((T - t) ** 3) / 6
                        q_dot = v1 + j_max * 0.5 * (T - t) ** 2
                        q_ddot = -j_max * (T - t)
                        #q_dddot = j_max

                #res = [sigma * q, sigma * q_dot, sigma * q_ddot, sigma * q_dddot]

                return sigma * q, sigma * q_dot, sigma * q_ddot