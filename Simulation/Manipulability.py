import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics


class Manipulability:
        def __init__(self):
                self.fk = ForwardKinematics()

        
        def manipulability(self, q):
                J = self.fk.jacobian_body(q)
                
                return np.sqrt(max(0, np.linalg.det(J@J.T)))
        
        def manipulability_gradient(self, q):
                
                w0 = self.manipulability(q)
                h = 0.00001
                grad = []
                for i in range(7):
                        dq = np.zeros(7)
                        dq[i] = h
                        grad.append((self.manipulability(q+dq) - w0)/h)
                        
                return np.array(grad)
        
        
        def dir_manipulability_gradient(self, q, dir):
                
                J = self.fk.jacobian(q)
                H = self.fk.hessian(q)
                W = np.diag(dir)
                
                b = np.linalg.inv(J@J.T + np.eye(6)*0.05)
                
                Jm = np.zeros(7)
                for i in range(7):
                        c = W @ J @ H[:,:,i].T
                        Jm[i] = (c.flatten("F").T)@b.flatten("F")
                        
                return Jm
        
        
        def dir_manipulability(self, q, v):
                Jb = self.fk.jacobian_body(q)
                U, S, V = np.linalg.svd(Jb@Jb.T)
                
                sum_w = 0
                for i in range(6):
                        sum_w += abs(S[i]*(np.dot(v, U[:,i])))
                        
                return sum_w
        
#         def dir_manipulability_gradient(self, q):
                
#                 v = np.array([1,0,0,0,0,1])
#                 w0 = self.dir_manipulability(q, v)
#                 h = 0.0001
#                 grad = []
#                 for i in range(7):
#                         dq = np.zeros(7)
#                         dq[i] = h
#                         grad.append((self.dir_manipulability(q+dq, v) - w0)/h)
                
#                 return np.array(grad)
        
        
        