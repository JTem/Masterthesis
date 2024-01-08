import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics

class Manipulability:
        
        # Initialize the Manipulability object with a given type of kinematics model
        def __init__(self, fk_type = "normal"):
                self.fk = ForwardKinematics(fk_type)
                self.dof = self.fk.dof
                
                
        # Calculate the yoshikawa manipulability measure for a given joint configuration
        def manipulability(self, q):
            
                J = self.fk.getBodyJacobian(q)
                
                return np.sqrt(max(0, np.linalg.det(J@J.T)))
        
        
        # Calculate the gradient of manipulability with respect to joint variables
        def manipulability_gradient(self, q):
                
                # Current manipulability index
                w0 = self.manipulability(q)
                h = 0.00001 # Current manipulability index
                grad = []
                
                for i in range(self.dof):
                        dq = np.zeros(self.dof)
                        dq[i] = h
                        
                        # Numerically approximate the gradient using finite difference
                        grad.append((self.manipulability(q+dq) - w0)/h)
                        
                # Return the gradient as an array
                return np.array(grad)
        
        
        # Calculate the directional manipulability gradient
        def dir_manipulability_gradient(self, q, direction):
                
                # Calculate the body Jacobian matrix for the joint configuration
                J = self.fk.getBodyJacobian(q)
                
                # Calculate the Hessian matrix for the joint configuration
                H = self.fk.getHessian(q)
                
                # Create a diagonal matrix from the direction array
                W = np.diag(direction)
                
                 # Calculate the inverse of JJ^T with regularization term for numerical stability
                b = np.linalg.inv(J@J.T + np.eye(6)*0.01)
                
                Jm = np.zeros(self.dof)
                for i in range(self.dof):
                        # Calculate the contribution of the ith joint to the manipulability gradient
                        c = W @ J @ H[:,:,i].T
                        Jm[i] = (c.flatten("F").T)@b.flatten("F")
                        
                # Return the directional manipulability gradient
                return Jm
        

        
        
        