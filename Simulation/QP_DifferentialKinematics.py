import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.Manipulability import Manipulability

import osqp
import scipy.sparse as sp
import scipy

class QP_DifferentialKinematics:
        
        def __init__(self, fk_type = "normal", method = "qp",  verbose = False):
                # Initialize forward kinematics and manipulability objects based on the specified type.
                self.fk = ForwardKinematics(fk_type)
                self.mp = Manipulability(fk_type)
                self.fk_type = fk_type
                self.method = method
                
                # Define settings for the OSQP optimizer.
                self.osqp_settings = {
                    'alpha':0.9,
                    'verbose':False,
                    'max_iter': 1000,
                    'eps_abs': 1e-2,
                    'eps_rel': 1e-2,
                    'check_termination': 75,
                }

                # Set the number of degrees of freedom (DOF) and the dimension of the Jacobian.
                self.dof = self.fk.dof
                self.dim_jac = 8
                
                # Initialize the constraint matrix with specific dimensions and initial values.
                dim1 = 3*self.dof + self.dim_jac + 1
                dim2 = 2*self.dof + 1
                self.ConstraintMatrix = np.zeros((dim1, dim2))
                
                # Fill specific parts of the constraint matrix with initial values.
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof+ 1, :self.dof+1] = np.eye(self.dof + 1)
                self.ConstraintMatrix[:self.dim_jac, :self.dof] = 2*np.ones((8, self.dof))
                self.ConstraintMatrix[:self.dim_jac, 2*self.dof:2*self.dof + 1] = -3*np.ones((8,1))
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof, :self.dof] = np.eye(self.dof)
                self.ConstraintMatrix[self.dim_jac:self.dim_jac + self.dof, self.dof:2*self.dof] = -0.005*np.eye(self.dof)
                self.ConstraintMatrix[self.dim_jac + self.dof:self.dim_jac + 3*self.dof + 1, :2*self.dof + 1] = np.eye(2*self.dof + 1)
                
                # Initialize weights and matrices for position and velocity gradients and for the quadratic program.
                self.Ws = 250 # timescaling weight
                self.Wv = 4.0 # velocity weight
                self.Wa = .0002 # acceleration weight
                
                # Set weights for position and velocity gradients, and initialize the P matrix, based on fk_type.
                if self.fk_type == "extended":
                        self.weight_pos_gradient = np.diag([0.1, 2, 0.5, 0.001, 2, 0.001, 0.001, 5])
                        self.weight_vel_gradient = np.diag([3, 3.0, 3.0, 1, 1, 1, 1, 3])
                        
                        # Define the P matrix for the quadratic programming problem in 'extended' mode.
                        self.P = sp.csc_matrix(self.Wv*np.diag([1, 1, 2, 0.1, 0.1, 0.1, 0.1, 0.1, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, 0.01*self.Wa/self.Wv, self.Ws/self.Wv]))
                else:
                        # Define weight matrices and P matrix for the default fk_type.
                        self.weight_pos_gradient = np.diag([0.1, 2.0, 2.0, 0.001, 0.5, 0.001, 0.001])
                        self.weight_vel_gradient = np.diag([3, 3.0, 3.0, 1, 1, 1, 1])
                        self.P = sp.csc_matrix(self.Wv*np.diag([1, 1, 2, 0.1, 0.1, 0.1, 0.1, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Wa/self.Wv, self.Ws/self.Wv]))
                
                # Initialize the gradient vector and set its last element.
                self.gradient = np.zeros(dim2)
                self.gradient[-1] = -2.0*self.Ws
                
                # Initialize the upper and lower bounds for the optimization problem.
                self.upper_bound = np.zeros(dim1)
                self.lower_bound = np.zeros(dim1)
                
                # Define joint limits and velocity limits.
                self.joint_limits = np.ones(self.dof)*3.14 # Setting a default value for all joints
                
                # Custom limits for specific joints.
                self.joint_limits[1] = np.pi/180.0*120.0
                self.joint_limits[3] = np.pi/180.0*150.0
                
                # Set the velocity limits for the joints.
                self.velocity_limits = np.ones(self.dof)*np.pi/180.0*120.0
                
                # Define the upper and lower bounds for the problem.
                self.upper_bound[self.dim_jac:self.dim_jac + self.dof] = np.ones(self.dof)*5
                self.upper_bound[self.dim_jac + self.dof:self.dim_jac + 2*self.dof] = self.velocity_limits
                self.upper_bound[self.dim_jac + 2*self.dof:self.dim_jac + 3*self.dof] = np.ones(self.dof)*np.inf
                self.upper_bound[-1] = 1.0 # Upper bound for the time scaling variable
                
                self.lower_bound[self.dim_jac:self.dim_jac + self.dof] = np.ones(self.dof)*5
                self.lower_bound[self.dim_jac + self.dof:self.dim_jac + 2*self.dof] = -self.velocity_limits
                self.lower_bound[self.dim_jac + 2*self.dof:self.dim_jac + 3*self.dof] = -np.ones(self.dof)*np.inf
                self.lower_bound[-1] = 0.01 # Lower bound for the time scaling variable
                
                # If verbose mode is enabled, print the initialized matrices and vectors for debugging.
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
                
                # Update the first block of rows in the constraint matrix with the Jacobian.
                # This part of the matrix is used to constrain the motion according to the Jacobian.
                Aconstraint[:self.dim_jac, :self.dof] = J
                
                # Update the same block of rows with the negative of the reference vector.
                # This part of the matrix is used to align the motion with the reference velocity.
                Aconstraint[:self.dim_jac, 2*self.dof:2*self.dof + 1] = -ref
                
                # Return the updated constraint matrix.
                return Aconstraint
        
        
        def vel_damper(self, q, q_min, q_max):
                
                # Initialize a gradient vector for velocity damping, with zeros.
                grad = np.zeros(self.dof)
                
                # Define a threshold to start damping when the joint angle is within this percentage of its limit.
                thresh = 0.8
                for i in range(self.dof):
                        # Damping when the joint angle is approaching the maximum limit.
                        if q[i] > thresh*q_max[i]:
                                grad[i] = (q[i] - thresh*q_max[i])/(abs(q[i] - q_max[i]) + 0.001)
                        
                        # Damping when the joint angle is approaching the minimum limit.
                        if q[i] < thresh*q_min[i]:
                                grad[i] = (q[i] - thresh*q_min[i])/(abs(q[i] - q_min[i]) + 0.001)
                                
                # Return the gradient vector for velocity damping.        
                return grad
        
            
        def updateGradient(self, q, q_dot, direction):
                
                gradient = np.zeros(2*self.dof + 1)
                
                # Update the first part of the gradient based on the chosen method
                if self.method == "qp":
                        # For the 'qp' method, calculate the directional manipulability gradient and update the gradient.
                        gradient[:self.dof] -= 2.0*self.mp.dir_manipulability_gradient(q, direction)
                if self.method == "qp_yoshikawa":
                        # For the 'qp_yoshikawa' method, calculate the manipulability gradient and update the gradient.
                        gradient[:self.dof] -= 2.0*self.mp.manipulability_gradient(q)
                        
                # Apply position-related gradient adjustments, for joint position minimization.
                gradient[:self.dof] += 0.1*self.weight_pos_gradient@q
                
                # Apply velocity damping based on joint limits.
                gradient[:self.dof] += 1.0*self.vel_damper(q, -self.joint_limits, self.joint_limits)
                
                # Apply velocity-related gradient adjustments.
                gradient[self.dof:2*self.dof] += 0.01*self.weight_vel_gradient@q_dot
                
                # Set the last element of the gradient.
                gradient[-1] = -2.0*self.Ws
                
                # Return the computed gradient.
                return gradient
        
        
        def updateLimits(self, lower_limit, upper_limit, q_dot_last):
                # Update the lower and upper limits for the joint velocities based on the last joint velocities.
                # This modification is done only for the elements corresponding to the joint velocities.
                lower_limit[self.dim_jac:self.dim_jac + self.dof] = q_dot_last
                upper_limit[self.dim_jac:self.dim_jac + self.dof] = q_dot_last
                
                return lower_limit, upper_limit
        
        
        def quadratic_program(self, q, q_dot, DQd, DQd_dot, direction):
                # Calculate forward kinematics to get current position and orientation of the end-effector
                x = self.fk.getFK(q)
                
                # Compute error as the vector difference between desired and current end-effector pose
                error = (DQd - x).asVector()
                
                # Compute the space Jacobian for the current joint angles
                J = self.fk.getSpaceJacobian8(q)
                
                # Calculate half Jacobian for use in the constraint matrix
                J_H = 0.5*DQd.as_mat_right()@J
                
                # Proportional control to calculate reference end-effector velocity
                kp = 20.0
                ref = (DQd_dot.asVector() + kp*error)
                
                # Update the constraint matrix for the quadratic program with the new Jacobian and reference velocity
                Acon = self.updateConstraintMatrix(self.ConstraintMatrix, J_H, ref)
                
                # Update the gradient of the optimization problem based on the current direction and state
                self.gradient = self.updateGradient(q, q_dot, direction)
                
                # Update the upper and lower bounds for the optimization problem
                self.lower_bound, self.upper_bound = self.updateLimits(self.lower_bound, self.upper_bound, q_dot)
                
                # Initialize and set up the quadratic programming problem
                prob = osqp.OSQP()
                prob.setup(self.P, self.gradient, sp.csc_matrix(Acon), self.lower_bound, self.upper_bound, **self.osqp_settings)
                
                # Solve the quadratic programming problem
                res = prob.solve()
                
                # Check if the problem was solved successfully
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status)
              
                # Extract the optimal joint velocities and additional scalar variable from the solution
                q_dot_ = res.x[:self.dof]
                s_ = res.x[-1]
                
                # Return the optimal joint velocities and the additional scalar variable
                return q_dot_, s_

        