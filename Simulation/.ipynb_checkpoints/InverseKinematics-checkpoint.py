import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics

class InverseKinematics:
        
        # Initialize the InverseKinematics object with a given type of kinematics model
        def __init__(self, fk_type = "normal"):
                self.T = None
                self.fk = ForwardKinematics(fk_type) # Forward kinematics object for calculating Jacobian
                self.dof = self.fk.dof
    
        # Get the inverse kinematics solution using dual quaternions
        def getIK_DQ(self, x_target, q_guess, damp):
       
                count = 0  # Initialize iteration counter
                q_sol = q_guess.copy() # Initialize the solution as the guess
                error_norm_list = [] # List to hold the error norms for each iteration
                I = np.eye(J.shape[1]) # Initialize the identity matrix for regularization
                while True:

                        x_act = self.fk.getFK(q_sol) # Calculate the current end-effector pose
                        
                        # Calculate the Jacobian matrix for the current pose and transform it to dual quaternion space
                        J = self.fk.getSpaceJacobian8(q_sol)
                        J_H = 0.5*x_act.as_mat_right()@J
                        
                        # error calculation
                        pos_error_norm = np.linalg.norm(x_target.getPosition() - x_act.getPosition())
                        orientation_error_norm = (x_target.real.inverse()*x_act.real).getAngle()
                        
                        error_norm = pos_error_norm + orientation_error_norm
                        
                        # Record error norm
                        error_norm_list.append(error_norm)
                        
                        # Check for convergence
                        if error_norm < 1.0e-6:
                                # Return the solution and success flag
                                return q_sol, error_norm_list, True
                        
                        # Calculate the damped pseudo-inverse of Jacobian
                        J_pinv = np.linalg.inv(J_H.T @ J_H + damp*error_norm*I) @ J_H.T  
                        
                        # Calculate the error in dual quaternion form
                        error = (x_target - x_act).asVector()
                        
                        # Calculate the change in joint variables
                        delta_q = J_pinv@error.flatten()
                        
                        # Update the joint variables
                        q_sol += delta_q
            
                        # Break the loop if too many iterations
                        if count > 20:
                                break

                        count += 1
                
                # Return the solution and failure flag if not converged
                return q_sol, error_norm_list, False
        
        
        def inverse_transformation_matrix(self, T):
                # Extract the rotation matrix R and position vector p from the transformation matrix T
                R = T[0:3, 0:3]
                p = T[0:3, 3]

                T_inv = np.eye(4) 
                # The inverse of the rotation part is the transpose of R
                T_inv[0:3, 0:3] = R.T
                # The inverse of the position part is -R.T * p
                T_inv[0:3, 3] = -R.T@p

                return T_inv

        
        def rot_logarithm(self, R):
                
                # Calculate the angle of rotation (theta) from the rotation matrix using the trace method
                theta = np.arccos(min(1.0, max(-1, 0.5*(np.trace(R) - 1.0))))
                
                # Handle the special cases based on the value of theta
                if abs(theta) < 1e-7:
                        # If theta is very small, return zero rotation
                        return theta, np.array([0,0,0])
                
                # If the trace of R is near -1, handle the singularity cases
                elif abs(np.trace(R) + 1) < 1e-7: 
                        
                        # Each of these conditions checks a different diagonal entry of R to avoid numerical instability
                        if abs(R[2,2] + 1) > 0.01:
                                return theta, np.array([R[0,2], R[1,2], 1 + R[2,2]])*(1.0/np.sqrt(2.0*(1.0 + R[2,2])))

                        if abs(R[1,1] + 1) > 0.01:
                                return theta, np.array([R[0,1], 1 + R[1,1], R[2,1]])*(1.0/np.sqrt(2.0*(1.0 + R[1,1])))

                        if abs(R[0,0] + 1) > 0.01:
                                return theta, np.array([1 + R[0,0], R[1,0], R[2,0]])*(1.0/np.sqrt(2.0*(1.0 + R[0,0])))
                        
                # For the general case where theta is neither too small nor too close to pi
                else:
                        # Calculate the skew-symmetric matrix of rotation vector
                        w_br = (1.0/(2.0*np.sin(theta)))*(R-R.T)
                        
                        # Return the rotation vector
                        return theta, np.array([w_br[2,1], w_br[0,2], w_br[1,0]])

                
        def skew_symmetric(self, v):
                # Generate and return the 3x3 skew-symmetric matrix from the 3D vector v
                return np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
        
        def cot(self, theta):
                # Calculate and return the cotangent of an angle theta
                return 1 / np.tan(theta)

        
        def matrix_logarithm(self, T):
                # Extract the rotation matrix R and position vector p from the transformation matrix T
                R = T[0:3, 0:3]
                p = T[0:3, 3]
                
                
                # Calculate the rotation angle and axis (theta, w) using rotation logarithm
                theta, w = self.rot_logarithm(R)

                if abs(theta) < 1e-7:
                        # If theta is very small, the logarithm is approximately the translation and zero rotation
                        return np.array([*w, *p])
                else:
                        # Calculate the skew-symmetric matrix of the rotation vector
                        skew = self.skew_symmetric(w)
                        
                        # Calculate the matrix G used in the expression for logarithm of T
                        G = np.eye(3)*theta + (1.0 - np.cos(theta))*skew + (theta - np.sin(theta)) * skew@skew
                        
                        # Calculate the equivalent velocity vector v from p using the matrix G
                        v = np.linalg.inv(G)@p
                        
                        # Return the twist vector (rotation and translation) scaled by theta
                        return np.array([*w, *v])*theta
        

                
        
        def getIK_classic(self, x_target, q_guess, damp):
       
                count = 0 # Initialize iteration counter
                q_sol = q_guess.copy() # Initialize the solution as the guess
                error_norm_list = [] # List to hold the error norms for each iteration
                I = np.eye(J.shape[0]) # Initialize the identity matrix for regularization
                while True:
                
                        # Calculate the current end-effector pose
                        x_act = self.fk.getFK(q_sol)
                        
                        # Calculate the space Jacobian for the current joint configuration
                        J = self.fk.getSpaceJacobian(q_sol)
                        
                        # Convert target and actual poses to transformation matrices
                        T_target = x_target.asTransformation()
                        T_act = x_act.asTransformation()
                        
                        # Calculate the transformation error as the matrix logarithm of the difference
                        dT = T_target@self.inverse_transformation_matrix(T_act)
                        log_error = self.matrix_logarithm(dT)
                        
                        # Calculate position and orientation errors
                        pos_error_norm = np.linalg.norm(x_target.getPosition() - x_act.getPosition())
                        orientation_error_norm = (x_target.real.inverse()*x_act.real).getAngle()
                        error_norm = pos_error_norm + orientation_error_norm
                        
                        # Record error norm
                        error_norm_list.append(error_norm)
                        
                        # Check for convergence
                        if error_norm < 1.0e-6:
                                # Return the solution and success flag
                                return q_sol, error_norm_list, True
                        
                        # Calculate the damped pseudo-inverse of Jacobian
                        J_pinv = J.T @ np.linalg.inv(J @ J.T + damp*error_norm*I)  
                
                        # Calculate the change in joint variables
                        delta_q = J_pinv@log_error
                        
                        # Update the joint variables
                        q_sol += delta_q
                        
                        # Break the loop if too many iterations
                        if count > 20:
                                break

                        count += 1
                
                # Return the solution and failure flag if not converged
                return q_sol, error_norm_list, False
        
