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
                    'alpha': 1.0,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-5
                }
                
                self.dof = 7
                self.dim_jac = 8
                self.Ts = 0.01
                
                self.B = self.Ts*np.eye(7)
                self.A = np.eye(7)
                
                dim1 = 2*self.dof + self.dim_jac + 3*self.dof
                dim2 = 3*self.dof + self.dim_jac
                
                self.ConstraintMatrix = np.zeros((dim1, dim2))
                
                self.ConstraintMatrix[:self.dof, :self.dof] = -np.eye(self.dof)
                self.ConstraintMatrix[:self.dof, self.dof:self.dof + self.dof] = self.B
                self.ConstraintMatrix[self.dof:self.dof + self.dim_jac, self.dof:self.dof + self.dof] = 2*np.ones((self.dim_jac, self.dof))
                self.ConstraintMatrix[self.dof:self.dof + self.dim_jac, 2*self.dof:2*self.dof + self.dim_jac] = np.eye(self.dim_jac)
                self.ConstraintMatrix[self.dof+ self.dim_jac:self.dof + self.dim_jac + self.dof, self.dof:self.dof + self.dof] = -np.eye(self.dof)
                self.ConstraintMatrix[(self.dof+ self.dim_jac):(self.dof + self.dim_jac) + self.dof, (2*self.dof + self.dim_jac):(2*self.dof + self.dim_jac) + self.dof] = self.Ts*np.eye(self.dof)
                
                self.ConstraintMatrix[(2*self.dof+ self.dim_jac):(2*self.dof + self.dim_jac) + self.dof, :self.dof] = np.eye(self.dof)
                self.ConstraintMatrix[(3*self.dof+ self.dim_jac):(3*self.dof + self.dim_jac) + self.dof, (self.dof):(self.dof) + self.dof] = np.eye(self.dof)
                self.ConstraintMatrix[(4*self.dof+ self.dim_jac):(4*self.dof + self.dim_jac) + self.dof, (2*self.dof + self.dim_jac):(2*self.dof + self.dim_jac) + self.dof] = np.eye(self.dof)
               # print(self.ConstraintMatrix)
                
                Qx = np.diag([5,25,25,5,0.1,0.1,0.1])
                #Qx = np.eye(self.dof)*5
                #Qu = np.eye(self.dof)*1
                Qu = np.diag([100,10,10,1,1,1,1])
                Qs = np.eye(self.dim_jac)*10_000_000
                Qdu = np.eye(self.dof)*0.000001
        
        
                hessian = scipy.linalg.block_diag(Qx, Qu, Qs, Qdu)
                #print(hessian)
                self.P = sp.csc_matrix(hessian)
                
                self.gradient = np.zeros(dim2)
                
                self.upper_bound = np.zeros(5*self.dof + self.dim_jac)
                self.lower_bound = np.zeros(5*self.dof + self.dim_jac)
                
                self.joint_limits = np.ones(7)*3.14
                self.velocity_limits = np.ones(7)*1.56
                self.acceleration_limits = np.ones(7)*np.inf
                
                self.upper_bound[2*self.dof + self.dim_jac:2*self.dof + self.dim_jac + self.dof] = self.joint_limits
                self.upper_bound[3*self.dof + self.dim_jac:3*self.dof + self.dim_jac + self.dof] = self.velocity_limits
                self.upper_bound[4*self.dof + self.dim_jac:4*self.dof + self.dim_jac + self.dof] = self.acceleration_limits
                
                self.lower_bound[2*self.dof + self.dim_jac:2*self.dof + self.dim_jac + self.dof] = -self.joint_limits
                self.lower_bound[3*self.dof + self.dim_jac:3*self.dof + self.dim_jac + self.dof] = -self.velocity_limits
                self.lower_bound[4*self.dof + self.dim_jac:4*self.dof + self.dim_jac + self.dof] = -self.acceleration_limits
                
                self.prob = osqp.OSQP()

                # Setup workspace and change settings
                self.prob.setup(self.P, self.gradient, sp.csc_matrix(self.ConstraintMatrix), self.lower_bound, self.upper_bound, **self.osqp_settings)

                

        def updateConstraintMatrix(self, J, Aconstraint):
                Aconstraint[self.dof:self.dof + self.dim_jac, self.dof:self.dof + self.dof] = J
                return Aconstraint
        
        
        def updateBounds(self, lower_bound, upper_bound, xk, u0, ref):
                lower_bound[:self.dof] = -xk
                lower_bound[self.dof:self.dof + self.dim_jac] = ref
                lower_bound[self.dof + self.dim_jac:self.dof + self.dim_jac + self.dof] = -u0
                
                upper_bound[:self.dof] = -xk
                upper_bound[self.dof:self.dof + self.dim_jac] = ref
                upper_bound[self.dof + self.dim_jac:self.dof + self.dim_jac + self.dof] = -u0
                
                return lower_bound, upper_bound
        
        
        def quadratic_program(self, q, q_dot, DQd, DQd_dot):
                
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
     
                kp = 60.0

                # Define constraint matrix and bounds
                ref = (DQd_dot.asVector() + kp*error).flatten()
                
                Acon = self.updateConstraintMatrix(J_H, self.ConstraintMatrix)
                self.lower_bound, self.upper_bound = self.updateBounds(self.lower_bound, self.upper_bound, q, q_dot, ref)
                
                self.gradient[self.dof:self.dof + self.dof] = self.dk.dir_manipulability_gradient2(q)
                #print(self.gradient)
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(self.P, -2*self.gradient, sp.csc_matrix(Acon), self.lower_bound, self.upper_bound, **self.osqp_settings)

                #Solve problem
                res = prob.solve()
                
                if res.info.status != 'solved':
                        # print(Acon)
                        # print(self.ConstraintMatrix)
                        print("The problem is ", res.info.status )                        
                        #return None
              
                # The problem was feasible (or possibly optimal)
                q_dot_ = res.x[self.dof:self.dof + self.dof]
                
                return q_dot_

        