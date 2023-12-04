import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from ModelPredictiveControl.ModelPredictiveControl import ModelPredictiveControl
import osqp
import scipy.sparse as sp

class MPC_DifferentialKinematics:
        
        def __init__(self, trajectory, N, Nu, dof, Ts0, Ts_lin_fact, Ts_quat_fact, weight_x, weight_u, weight_s, weight_du, joint_limit, velocity_limit, acceleration_limit):
                self.forward_kinematics = ForwardKinematics()
                
                self.A = np.zeros((dof,dof))
                self.B = np.eye(dof)
                
                self.joint_limits = joint_limit
                self.velocity_limits = velocity_limit
                self.acceleration_limits = acceleration_limit
                
                mpc = ModelPredictiveControl(N, Nu, dof)
                self.dt_vector = mpc.computeDeltaTimeVector(Ts, N, Ts_lin_fact, Ts_quat_fact)
                
                Qx = np.eye(dof)*weight_x
                Qu = np.eye(dof)*weight_u
                Qs = np.eye(8)*weight_s
                Qdu = np.eye(dof)*weight_du
                self.Q = mpc.initializeHessian(Qx, Qu, Qs, Qdu)
        
                self.A_constraints = mpc.initializeConstraintMatrix(A, B, J_list, dt_vector)
                self.lower_constraint, self.upper_constraint = mpc.initializeConstraintVectors(Ts, x0, A, u0, joint_limits, velocity_limits, acceleration_limits)
                
                self.prob = osqp.OSQP()
                
                self.osqp_settings = {
                    'alpha': 1.0,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-4
                }

        def seed(q):
                x = self.forward_kinematics.forward_kinematics(q)
                J = self.forward_kinematics.jacobian(q)
                J_H = 0.5*x.as_mat_right()@J
                
                J_list = []
                ref_list = []
                for i in range(Nu):
                        J_list.append(J)
                        _, dq_dot, _ = self.trajectory.evaluateDQ(self.dt_vector[i])
                        ref_list.append(dq_dot.asVector().flatten())
                                        
                self.A_constraints = mpc.updateConstraintMatrix(self.A_constraints, J_list)
                self.lower_constraint, self.upper_constraint = mpc.updateConstraintVector(self.lower_constraint, self.upper_constraint, q, np.zeros(dof), ref_list)
                

                # Setup workspace and change settings
                self.prob.setup(self.Q, None, self.A_constraints, self.lower_constraint, self.upper_constraint, **self.osqp_settings)
                                        
                # Solve problem
                res = prob.solve()
                
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status )                        
              
                # The problem was feasible (or possibly optimal)
                x = res.x
                        
                
        def update(self, q, time):
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J_list = []
                ref_list = []
                for i in range(Nu):
                        xi = self.forward_kinematics.forward_kinematics(q)
                        J_list.append(J)
                        _, dq_dot, _ = self.trajectory.evaluateDQ(self.dt_vector[i])
                        ref_list.append(dq_dot.asVector().flatten())
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                #J_H_reg = J_H + 0.0001 * np.eye(J_H.shape[0])
                kp = 20.0

                # Define OSQP data
                P = sp.csc_matrix(np.eye(7))   # Quadratic term
                q = np.zeros(7)                # Linear term

                # Define constraint matrix and bounds
                A = sp.csc_matrix(J_H)
                l = (DQd_dot.asVector() + kp*error).flatten()
                u = l  # Equality constraint

    

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

