import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.Manipulability import Manipulability

class DifferentialKinematics:
        # Initialize the DifferentialKinematics object with forward kinematics and manipulability
        def __init__(self, fk_type = "classic"):
                
                self.fk_type = fk_type
                self.fk = ForwardKinematics(self.fk_type) # Forward kinematics object
                self.mp = Manipulability(fk_type) # Manipulability object for calculating gradients
                self.dof = self.fk.dof
                self.gradient = np.zeros(self.dof)
        
        # Calculate the joint velocities required to achieve a desired end-effector velocity using dual quaternions
        def differential_kinematics_DQ(self, q, q_dot, DQd, DQd_dot):
                # Calculate the current pose of the end-effector
                x = self.fk.getFK(q)
                
                # Calculate the error between desired and current dual quaternion poses
                error = (DQd - x).asVector()
                
                # Calculate the 8-dimensional Jacobian in the space frame
                J = self.fk.getSpaceJacobian8(q)
                
                # Calculate the dual quaternion Jacobian
                J_H = 0.5*DQd.as_mat_right()@J
                
                # Calculate the pseudo-inverse of the Jacobian for solving least squares problem
                pinv = np.linalg.pinv(J_H)
                
                # Proportional gain for error correction in task space
                kp = 20
                
                # Calculate the desired velocity in task space, combining feedforward and feedback terms
                vel = DQd_dot.asVector() + kp*error
                
                # Calculate the manipulability gradient to avoid singular configurations
                self.gradient = self.mp.manipulability_gradient(q)
                
                # Calculate the joint velocity command, incorporating manipulability maximization
                q_dot_ = pinv@vel.flatten() + 5.0*(np.eye(self.dof)-pinv@J_H)@self.gradient
                
                # Return the joint velocity command and a scalar time scale (fixed at 1 here)
                return q_dot_.flatten(), 1
        
        
               