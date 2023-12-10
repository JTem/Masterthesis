import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics

class Manipulability:
    
        def __init__(self, fk_type = "normal"):
                self.fk = ForwardKinematics(fk_type)
                self.dof = self.fk.dof
                
        
        def manipulability(self, q):
            
                J = self.fk.getBodyJacobian(q)
                
                return np.sqrt(max(0, np.linalg.det(J@J.T)))
        
        
        def manipulability_gradient(self, q):
                
                w0 = self.manipulability(q)
                h = 0.00001
                grad = []
                for i in range(self.dof):
                        dq = np.zeros(self.dof)
                        dq[i] = h
                        grad.append((self.manipulability(q+dq) - w0)/h)
                        
                return np.array(grad)
        
        
        def dir_manipulability_gradient(self, q, direction):
                
                J = self.fk.getBodyJacobian(q)
                H = self.fk.getHessian(q)
                W = np.diag(direction)
                
                b = np.linalg.inv(J@J.T + np.eye(6)*0.05)
                
                Jm = np.zeros(self.dof)
                for i in range(self.dof):
                        c = W @ J @ H[:,:,i].T
                        Jm[i] = (c.flatten("F").T)@b.flatten("F")
                        
                return Jm
        
        
        def dir_manipulability_projection(self, q, direction):
            
                J = self.fk.getGeometricJacobian(q)
                U, S, V = np.linalg.svd(J@J.T)
                
                sum_w = 0
                for i in range(6):
                        sum_w += abs(S[i]*(np.dot(direction, U[:,i])))
                        
                return sum_w
        
        def dir_manipulability_gradient_projection(self, q, direction):
                
                w0 = self.dir_manipulability_projection(q, direction)
                h = 0.000001
                grad = []
                for i in range(7):
                        dq = np.zeros(7)
                        dq[i] = h
                        grad.append((self.dir_manipulability_projection(q+dq, direction) - w0)/h)
                
                return np.array(grad)
        
        
        