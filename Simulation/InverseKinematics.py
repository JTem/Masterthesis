import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics

class InverseKinematics:
        
        def __init__(self, fk_type = "normal"):
                self.T = None
                self.fk = ForwardKinematics(fk_type)
                self.dof = self.fk.dof
    
        def getIK_DQ(self, x_target, q_guess, damp):
       
                count = 0
                q_sol = q_guess.copy()
                error_norm_list = []
                while True:

                        x_act = self.fk.getFK(q_sol)

                        J = self.fk.getSpaceJacobian8(q_sol)
                        J_H = 0.5*x_act.as_mat_right()@J
                        
                        I = np.eye(J.shape[1])
                        
                        # error calculation
                        pos_error_norm = np.linalg.norm(x_target.getPosition() - x_act.getPosition())
                        orientation_error_norm = (x_target.real.inverse()*x_act.real).getAngle()
                        
                        error_norm = pos_error_norm + orientation_error_norm
                        
                        error_norm_list.append(error_norm)
                        
                        if error_norm < 1.0e-6:
                                return q_sol, error_norm_list, True
                        
                        
                        J_pinv = np.linalg.inv(J_H.T @ J_H + damp*error_norm*I) @ J_H.T  

                        error = (x_target - x_act).asVector()

                        delta_q = J_pinv@error.flatten()

                        q_sol += delta_q
            

                        if count > 20:
                                break

                        count += 1

                return q_sol, error_norm_list, False
        
        
        def inverse_transformation_matrix(self, T):

                R = T[0:3, 0:3]
                p = T[0:3, 3]

                T_inv = np.eye(4) 
                T_inv[0:3, 0:3] = R.T
                T_inv[0:3, 3] = -np.dot(R.T, p) 

                return T_inv

        def rot_logarithm(self, R):

                theta = np.arccos(min(1.0, max(-1, 0.5*(np.trace(R) - 1.0))))

                if abs(theta) < 1e-7:
                        return theta, np.array([0,0,0])

                elif abs(np.trace(R) + 1) < 1e-7:

                        if abs(R[2,2] + 1) > 0.01:
                                return theta, np.array([R[0,2], R[1,2], 1 + R[2,2]])*(1.0/np.sqrt(2.0*(1.0 + R[2,2])))

                        if abs(R[1,1] + 1) > 0.01:
                                return theta, np.array([R[0,1], 1 + R[1,1], R[2,1]])*(1.0/np.sqrt(2.0*(1.0 + R[1,1])))

                        if abs(R[0,0] + 1) > 0.01:
                                return theta, np.array([1 + R[0,0], R[1,0], R[2,0]])*(1.0/np.sqrt(2.0*(1.0 + R[0,0])))
                else:
                        w_br = (1.0/(2.0*np.sin(theta)))*(R-R.T)
                        return theta, np.array([w_br[2,1], w_br[0,2], w_br[1,0]])

        def skew_symmetric(self, v):

                return np.array([[0, -v[2], v[1]],
                             [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
        
        def cot(self, theta):
                return 1 / np.tan(theta)

        def matrix_logarithm(self, T):

                R = T[0:3, 0:3]
                p = T[0:3, 3]
           
                theta, w = self.rot_logarithm(R)

                if abs(theta) < 1e-7:
                        return np.array([*w, *p])
                else:
                        skew = self.skew_symmetric(w)
                      
                        G = np.eye(3)*theta + (1.0 - np.cos(theta))*skew + (theta - np.sin(theta)) * skew@skew
                        
                        v = np.linalg.inv(G)@p

                        return np.array([*w, *v])*theta
        

                
        
        def getIK_classic(self, x_target, q_guess, damp):
       
                count = 0
                q_sol = q_guess.copy()
                error_norm_list = []
      
                while True:

                        x_act = self.fk.getFK(q_sol)

                        J = self.fk.getSpaceJacobian(q_sol)
                
                        I = np.eye(J.shape[0])
                        
                        T_target = x_target.asTransformation()
                        T_act = x_act.asTransformation()

                        dT = T_target@self.inverse_transformation_matrix(T_act)
                        log_error = self.matrix_logarithm(dT)
                        
                
                        pos_error_norm = np.linalg.norm(x_target.getPosition() - x_act.getPosition())
                        orientation_error_norm = (x_target.real.inverse()*x_act.real).getAngle()
          
                        error_norm = pos_error_norm + orientation_error_norm
                        
                        error_norm_list.append(error_norm)
                        
                        if error_norm < 1.0e-6:
                                return q_sol, error_norm_list, True
                        
                        
                        J_pinv = J.T @ np.linalg.inv(J @ J.T + damp*error_norm*I)  
                
                        delta_q = J_pinv@log_error

                        q_sol += delta_q
              
                        if count > 20:
                                break

                        count += 1

                return q_sol, error_norm_list, False
        
