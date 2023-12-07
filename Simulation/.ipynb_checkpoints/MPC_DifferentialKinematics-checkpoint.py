import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from ModelPredictiveControl.ModelPredictiveControl import ModelPredictiveControl
import osqp
import scipy.sparse as sp

class MPC_DifferentialKinematics:
        
        def __init__(self, N, Nu, dof, Ts0, Ts_lin_fact, Ts_quat_fact, weight_x, weight_u, weight_s, weight_du, joint_limit, velocity_limit, acceleration_limit):
                self.forward_kinematics = ForwardKinematics()
                
                self.N = N
                self.Nu = Nu
                
                self.A = np.zeros((dof,dof))
                self.B = np.eye(dof)
                
                self.joint_limits = joint_limit
                self.velocity_limits = velocity_limit
                self.acceleration_limits = acceleration_limit
                
                self.is_seeded = False
                
                self.mpc = ModelPredictiveControl(N, Nu, dof)
                self.dt_vector = self.mpc.computeDeltaTimeVector(Ts0, N, Ts_lin_fact, Ts_quat_fact)
                print(self.dt_vector)
                
                Qx = np.eye(dof)*weight_x
                Qu = np.eye(dof)*weight_u
                Qs = np.eye(8)*weight_s
                Qdu = np.eye(dof)*weight_du
                self.P = self.mpc.initializeHessian(Qx, Qu, Qs, Qdu)
                
                self.gradient = self.mpc.initializeGradient()
                
                self.A_constraints = self.mpc.initializeConstraintMatrix(self.A, self.B, self.dt_vector)
                self.lower_constraint, self.upper_constraint = self.mpc.initializeConstraintVectors(Ts0, np.zeros(7), self.A, np.zeros(7), joint_limit, velocity_limit, acceleration_limit)
                
                #self.prob = osqp.OSQP()
                
                self.osqp_settings = {
                    'alpha': 1.0,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-5
                }
                #self.prob.setup(self.Q, self.gradient, sp.csc_matrix(self.A_constraints), self.lower_constraint, self.upper_constraint, **self.osqp_settings)

        def seed(self, x0, u0, J, ref_list):
                self.is_seeded = True
                
                J_list = []
                for i in range(self.Nu):
                        J_list.append(J)
                                        
                self.A_constraints = self.mpc.updateConstraintMatrix(self.A_constraints, J_list)
                #print(self.A_constraints)
                self.lower_constraint, self.upper_constraint = self.mpc.updateConstraintVector(self.lower_constraint, self.upper_constraint, x0, u0, ref_list)
                
                #self.prob.setup(self.Q, self.gradient, sp.csc_matrix(self.A_constraints), self.lower_constraint, self.upper_constraint, **self.osqp_settings)
                #self.prob.update(q=self.gradient, l=self.lower_constraint, u=self.upper_constraint)       
                #self.prob.update(Px=self.Q.data, Ax=sp.csc_matrix(self.A_constraints).data)
                
                # Solve problem
                #res = self.prob.solve()
                
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(self.P, self.gradient, sp.csc_matrix(self.A_constraints), self.lower_constraint, self.upper_constraint, **self.osqp_settings)

                #Solve problem
                res = prob.solve()
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status )                        
                
                #print(res.x)
                return res.x
                        
                
        def update(self, x0, u0, J_list, ref_list):
                
                self.A_constraints = self.mpc.updateConstraintMatrix(self.A_constraints, J_list)
                
                self.lower_constraint, self.upper_constraint = self.mpc.updateConstraintVector(self.lower_constraint, self.upper_constraint, x0, u0, ref_list)
                
                A_csc = sp.csc_matrix(self.A_constraints)

                #self.prob.update(q=self.gradient, l=self.lower_constraint, u=self.upper_constraint)
                #self.prob.update(Px=self.Q.data, Ax=sp.csc_matrix(self.A_constraints).data)
                # Solve problem
                #res = self.prob.solve()
                
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(self.P, self.gradient, A_csc, self.lower_constraint, self.upper_constraint, **self.osqp_settings)

                #Solve problem
                res = prob.solve()
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status )                        

                return res.x

