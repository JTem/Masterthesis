import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
import osqp
import scipy.sparse as sp

class MPC_DifferentialKinematics:
        def __init__(self):
                self.forward_kinematics = ForwardKinematics()

                self.osqp_settings = {
                    'alpha': 1.0,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-4
                }

        
        def quadratic_program_1(self, q, q_dot, DQd, DQd_dot):
                
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                kp = 20.0

                # Define OSQP data
                P = sp.csc_matrix(np.eye(7))   # Quadratic term
                q = np.zeros(7)                # Linear term

                # Define constraint matrix and bounds
                A = sp.csc_matrix(J_H)
                l = (DQd_dot.asVector() + kp*error).flatten()
                u = l  # Equality constraint

                # Create an OSQP object
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(P, q, A, l, u, **self.osqp_settings)

                # Solve problem
                res = prob.solve()
                
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status )                        
                        return self.q_dot_last_
              
                # The problem was feasible (or possibly optimal)
                q_dot_ = res.x
                self.q_dot_last_ = q_dot_.copy()
                return q_dot_


