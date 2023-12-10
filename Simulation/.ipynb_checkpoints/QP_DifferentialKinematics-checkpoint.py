import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from Simulation.Manipulability import Manipulability

import osqp
import scipy.sparse as sp
import scipy

class QP_DifferentialKinematics:
        
        def __init__(self, fk_type = "normal"):
                self.fk = ForwardKinematics(fk_type)
                self.mp = Manipulability(fk_type)
                
                
                self.osqp_settings = {
                    'alpha': 1,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-3,
                    'eps_rel': 1e-3,
                    'check_termination': 25,
                }

                
                self.dof = self.fk.dof
                self.dim_jac = 8
                
                dim1 = self.dof + self.dim_jac
                dim2 = self.dof
                
                self.ConstraintMatrix = np.zeros((dim1, dim2))
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof, :self.dof] = np.eye(self.dof)
                
                self.weight_pos_gradient = np.diag([10, 10, 15.0, 0.1, 0.001, 0.001, 0.001])
                self.P = sp.csc_matrix(0.5*np.diag([1,1,5,0.1,0.1,0.1,0.1]))
                
                self.gradient = np.zeros(dim2)
                
                self.upper_bound = np.zeros(self.dof + self.dim_jac)
                self.lower_bound = np.zeros(self.dof + self.dim_jac)
                
                self.velocity_limits = np.ones(7)*2.56
                
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
                gradient[:self.dof] -= 1.0*self.mp.dir_manipulability_gradient(q, direction)
                #gradient[:self.dof] -= 1.0*self.mp.dir_manipulability_gradient_projection(q, direction)
                gradient[:self.dof] += 0.5*self.weight_pos_gradient@q
                return gradient
        
        
        def quadratic_program(self, q, q_dot, DQd, DQd_dot, direction):
                
                x = self.fk.getFK(q)
                
                error = (DQd - x).asVector()
                
                J = self.fk.getSpaceJacobian8(q)
                J_H = 0.5*DQd.as_mat_right()@J
            
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

        