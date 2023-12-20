import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.Manipulability import Manipulability

import osqp
import scipy.sparse as sp
import scipy

class QP_DifferentialKinematics:
        
        def __init__(self, fk_type = "normal", verbose = False):
                self.fk = ForwardKinematics(fk_type)
                self.mp = Manipulability(fk_type)
                self.fk_type = fk_type
                
                self.osqp_settings = {
                    'alpha':0.9,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-2,
                    'eps_rel': 1e-2,
                    'check_termination': 75,
                }

                
                self.dof = self.fk.dof
                self.dim_jac = 8
                
                dim1 = 3*self.dof + self.dim_jac + 1
                dim2 = 2*self.dof + 1
                
                self.ConstraintMatrix = np.zeros((dim1, dim2))
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof+ 1, :self.dof+1] = np.eye(self.dof + 1)
                
                #self.ConstraintMatrix[:self.dim_jac, :self.dof] = 2*np.ones((8, 7))
                #self.ConstraintMatrix[:self.dim_jac, 2*self.dof:2*self.dof + 1] = -3*np.ones((8,1))
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof, :self.dof] = np.eye(self.dof)
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof, self.dof:2*self.dof] = -0.005*np.eye(self.dof)
                self.ConstraintMatrix[self.dim_jac + self.dof:self.dim_jac + 3*self.dof + 1, :2*self.dof + 1] = np.eye(2*self.dof + 1)
                
                
                self.Ws = 45
                self.Wv = 2.5
                self.Wa = .001 # acceleration weight
                
                if self.fk_type == "extended":
                        self.weight_pos_gradient = np.diag([0.1, 2, 0.5, 0.001, 2, 0.001, 0.001, 5])
                        self.weight_vel_gradient = np.diag([3, 3.0, 3.0, 1, 1, 1, 1, 3])
                        self.P = sp.csc_matrix(self.Wv*np.diag([1, 1, 2, 0.1, 0.1, 0.1, 0.1, 0.1, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, 0.01*self.Wa/self.Wv, self.Ws/self.Wv]))
                else:
                        self.weight_pos_gradient = np.diag([0.1, 2.0, 2.0, 0.001, 0.5, 0.001, 0.001])
                        self.weight_vel_gradient = np.diag([3, 3.0, 3.0, 1, 1, 1, 1])
                        self.P = sp.csc_matrix(self.Wv*np.diag([1, 1, 2, 0.1, 0.1, 0.1, 0.1, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Ws/self.Wv]))
                
                self.gradient = np.zeros(dim2)
                self.gradient[-1] = -2.0*self.Ws
                
                self.upper_bound = np.zeros(dim1)
                self.lower_bound = np.zeros(dim1)
                
                self.joint_limits = np.ones(self.dof)*3.14
                self.joint_limits[1] = np.pi/180.0*120.0
                self.joint_limits[3] = np.pi/180.0*150.0
                
                self.velocity_limits = np.ones(self.dof)*np.pi/180.0*120.0

                self.upper_bound[self.dim_jac:self.dim_jac + self.dof] = np.ones(self.dof)*5
                self.upper_bound[self.dim_jac + self.dof:self.dim_jac + 2*self.dof] = self.velocity_limits
                self.upper_bound[self.dim_jac + 2*self.dof:self.dim_jac + 3*self.dof] = np.ones(self.dof)*np.inf
                self.upper_bound[-1] = 1.0
                
                self.lower_bound[self.dim_jac:self.dim_jac + self.dof] = np.ones(self.dof)*5
                self.lower_bound[self.dim_jac + self.dof:self.dim_jac + 2*self.dof] = -self.velocity_limits
                self.lower_bound[self.dim_jac + 2*self.dof:self.dim_jac + 3*self.dof] = -np.ones(self.dof)*np.inf
                self.lower_bound[-1] = 0.0
                
                if verbose:
                        print("Weighting matrix P:")
                        print(self.P.toarray())
                        print("")
                        print("Constraint matrix A (with placeholders for Jacobian and reference):")
                        print(self.ConstraintMatrix)
                        print("")
                        print("the gradient (where the introduced gradients are not applied yet):")
                        print(self.gradient)
                        print("")
                        print("constant lower bound:")
                        print(self.lower_bound)
                        print("")
                        print("constant upper bound:")
                        print(self.upper_bound)

                        

        def updateConstraintMatrix(self, Aconstraint, J, ref):
                
                Aconstraint[:self.dim_jac, :self.dof] = J
                Aconstraint[:self.dim_jac, 2*self.dof:2*self.dof + 1] = -ref
                
                return Aconstraint
        
        
        
        def vel_damper(self, q, q_min, q_max):
                grad = np.zeros(self.dof)
                
                thresh = 0.8
                for i in range(self.dof):
                
                        if q[i] > thresh*q_max[i]:
                                grad[i] = (q[i] - thresh*q_max[i])/(abs(q[i] - q_max[i]) + 0.01)
                        
                        if q[i] < thresh*q_min[i]:
                                grad[i] = (q[i] - thresh*q_min[i])/(abs(q[i] - q_min[i]) + 0.01)
                        
                return grad
        
            
        def updateGradient(self, q, q_dot, direction):
                
                gradient = np.zeros(2*self.dof + 1)
                gradient[:self.dof] -= 1*self.mp.dir_manipulability_gradient(q, direction)
                gradient[:self.dof] += 2.0*self.weight_pos_gradient@q
                gradient[:self.dof] += 1.0*self.vel_damper(q, -self.joint_limits, self.joint_limits)
                
                gradient[self.dof:2*self.dof] += 0.001*self.weight_vel_gradient@q_dot
                
                gradient[-1] = -2.0*self.Ws
                
                return gradient
        
        
        def updateLimits(self, lower_limit, upper_limit, q_dot_last):
                
                lower_limit[self.dim_jac:self.dim_jac + self.dof] = q_dot_last
                upper_limit[self.dim_jac:self.dim_jac + self.dof] = q_dot_last
                
                return lower_limit, upper_limit
        
        
        def quadratic_program(self, q, q_dot, DQd, DQd_dot, direction):
                
                x = self.fk.getFK(q)
                
                error = (DQd - x).asVector()
                
                J = self.fk.getSpaceJacobian8(q)
                J_H = 0.5*DQd.as_mat_right()@J
            
                kp = 20.0
                ref = (DQd_dot.asVector() + kp*error)
                
                Acon = self.updateConstraintMatrix(self.ConstraintMatrix, J_H, ref)
              
                self.gradient = self.updateGradient(q, q_dot, direction)
                self.lower_bound, self.upper_bound = self.updateLimits(self.lower_bound, self.upper_bound, q_dot)
                
                prob = osqp.OSQP()
                prob.setup(self.P, self.gradient, sp.csc_matrix(Acon), self.lower_bound, self.upper_bound, **self.osqp_settings)
                res = prob.solve()
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status)
              
                q_dot_ = res.x[:self.dof]
                s_ = res.x[-1]
                
                return q_dot_, s_

        