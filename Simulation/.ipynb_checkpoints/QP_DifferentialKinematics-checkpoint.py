import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.Manipulability import Manipulability

import osqp
import scipy.sparse as sp
import scipy

class QP_DifferentialKinematics:
        
        def __init__(self, fk_type = "normal"):
                self.fk = ForwardKinematics(fk_type)
                self.mp = Manipulability(fk_type)
                
                
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
                
                dim1 = self.dof + self.dim_jac + 1
                dim2 = self.dof + 1
                
                self.ConstraintMatrix = np.zeros((dim1, dim2))
                
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof+ 1, :self.dof+1] = np.eye(self.dof + 1)
                
                #self.ConstraintMatrix[:self.dim_jac, :self.dof] = 3*np.ones((8, 7))
                #self.ConstraintMatrix[:self.dim_jac, self.dof:self.dof + 1] = -5*np.ones((8,1))
                
                self.Ws = 100
                self.Wv = 0.7
                self.weight_pos_gradient = np.diag([0.1, 2, 0.1, 0.001, 2, 0.001, 0.001])
                self.P = sp.csc_matrix(self.Wv*np.diag([1,1,5,0.1,0.1,0.1,0.1, self.Ws/self.Wv]))
                
                self.gradient = np.zeros(dim2)
                self.gradient[-1] = -self.Ws
                
                self.upper_bound = np.zeros(dim1)
                self.lower_bound = np.zeros(dim1)
                
                self.joint_limits = np.ones(self.dof)*3.14
                self.joint_limits[1] = np.pi/180.0*120.0
                self.joint_limits[3] = np.pi/180.0*150.0
                
                self.velocity_limits = np.ones(self.dof)*1.56
                self.upper_bound[self.dim_jac:self.dim_jac + self.dof] = self.velocity_limits
                self.upper_bound[-1] = 1.0
                
                self.lower_bound[self.dim_jac:self.dim_jac + self.dof] = -self.velocity_limits
                self.lower_bound[-1] = 0.05
                
                print(self.ConstraintMatrix)
                print(self.lower_bound)
                print(self.upper_bound)
                

        def updateConstraintMatrix(self, Aconstraint, J, ref):
                Aconstraint[:self.dim_jac, :self.dof] = J
                Aconstraint[:self.dim_jac, self.dof:self.dof + 1] = -ref
                return Aconstraint
        
        
#         def updateGradient(self, q, direction):
                
#                 self.gradient[:self.dof] = -1.0*self.mp.dir_manipulability_gradient(q, direction) + 0.2*self.weight_pos_gradient@q
        
        def vel_damper(self, q, q_min, q_max):
            grad = np.zeros(self.dof)
            
            for i in range(self.dof):
                
                if q[i] > 0.70*q_max[i]:
                        grad[i] = (q[i] - 0.85*q_max[i])/(abs(q[i] - q_max[i]) + 0.001)
                        
                if q[i] < 0.70*q_min[i]:
                        grad[i] = (q[i] - 0.85*q_min[i])/(abs(q[i] - q_min[i]) + 0.001)
                        
            return grad
            
        def updateGradient(self, q, direction):
                
                gradient = np.zeros(self.dof + 1)
                gradient[:self.dof] -= 1.0*self.mp.dir_manipulability_gradient(q, direction)
                #gradient[:self.dof] -= 1.0*self.mp.dir_manipulability_gradient_projection(q, direction)
                gradient[:self.dof] += 2*self.weight_pos_gradient@q
                gradient[:self.dof] += 0.05*self.vel_damper(q, -self.joint_limits, self.joint_limits)
                gradient[-1] = -self.Ws
                
                return gradient
        
        
        def quadratic_program(self, q, q_dot, DQd, DQd_dot, direction):
                
                x = self.fk.getFK(q)
                
                error = (DQd - x).asVector()
                
                J = self.fk.getSpaceJacobian8(q)
                J_H = 0.5*DQd.as_mat_right()@J
            
                kp = 20.0
                ref = (DQd_dot.asVector() + kp*error)
                
                Acon = self.updateConstraintMatrix(self.ConstraintMatrix, J_H, ref)
                #self.lower_bound, self.upper_bound = self.updateBounds(self.lower_bound, self.upper_bound, ref)
    
                self.gradient = self.updateGradient(q, direction)
                #print(self.gradient)
                prob = osqp.OSQP()
                prob.setup(self.P, self.gradient, sp.csc_matrix(Acon), self.lower_bound, self.upper_bound, **self.osqp_settings)
                res = prob.solve()
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status)
              
                q_dot_ = res.x[:self.dof]
                s_ = res.x[-1]
                
                return q_dot_, s_

        