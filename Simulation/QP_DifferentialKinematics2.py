import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
import osqp
import scipy.sparse as sp
import scipy

class QP_DifferentialKinematics:
        
        def __init__(self):
                self.forward_kinematics = ForwardKinematics()
                self.dk = DifferentialKinematics()
                
                self.osqp_settings = {
                    'alpha': 0.9,
                    'verbose': True,
                    'max_iter': 1000,
                    'eps_abs': 1e-1,
                    'eps_rel': 1e-1,
                    'check_termination': 100,
                }

                
                self.dof = 7
                self.dim_jac = 8
                
                dim1 = self.dof + self.dim_jac
                dim2 = self.dof
                
                self.ConstraintMatrix = np.zeros((dim1, dim2))
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof, :self.dof] = np.eye(self.dof)
                
                self.weight_pos_gradient = np.diag([15, 15, 25.0, 0.1, 0.001, 0.001, 0.001])
                self.P = sp.csc_matrix(np.diag([1,1,5,0.1,0.0001,0.0001,0.0001]))
                
                self.gradient = np.zeros(dim2)
                
                self.upper_bound = np.zeros(self.dof + self.dim_jac)
                self.lower_bound = np.zeros(self.dof + self.dim_jac)
                
                self.velocity_limits = np.ones(7)*1.56
                
                self.upper_bound[self.dim_jac:self.dim_jac + self.dof] = self.velocity_limits
                self.lower_bound[self.dim_jac:self.dim_jac + self.dof] = -self.velocity_limits
                
                

        def updateConstraintMatrix(self, J, Aconstraint):
                Aconstraint[:self.dim_jac, :self.dof] = J
                return Aconstraint
        
        
        def updateBounds(self, lower_bound, upper_bound, ref):
                lower_bound[:self.dim_jac] = ref
                upper_bound[:self.dim_jac] = ref
                
                return lower_bound, upper_bound
        
        
        def updateGradient(self, q, direction):
                
                gradient = np.zeros(self.dof)
                gradient[:self.dof] -= 2.0*self.dk.dir_manipulability_gradient2(q, direction)
                gradient[:self.dof] += 1*self.weight_pos_gradient@q
                return gradient
        
        
        def quadratic_program(self, q, q_dot, DQd, DQd_dot, direction):
                
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                J_H = 0.5*DQd.as_mat_right()@J
                
                # direction = (2.0*DQd.inverse()*DQd_dot).as6Vector().flatten()
                # if np.linalg.norm(direction) > 1e-6:
                #         direction = np.abs(direction)/np.linalg.norm(direction)
                
                print(direction)
                kp = 20.0
                ref = (DQd_dot.asVector() + kp*error).flatten()
                
                Acon = self.updateConstraintMatrix(J_H, self.ConstraintMatrix)
                self.lower_bound, self.upper_bound = self.updateBounds(self.lower_bound, self.upper_bound, ref)
                
                self.gradient = self.updateGradient(q, direction)
                prob = osqp.OSQP()

                prob.setup(self.P, self.gradient, sp.csc_matrix(Acon), self.lower_bound, self.upper_bound, **self.osqp_settings)
                res = prob.solve()
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status)
              
                q_dot_ = res.x
                
                return q_dot_

        