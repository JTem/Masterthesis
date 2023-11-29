import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics

class InverseKinematics:
        
        def __init__(self, min_lim, max_lim):
                self.T = None
                self.fk = ForwardKinematics()
                self.min_lim = min_lim
                self.max_lim = max_lim
    
        def getIK_DQ(self, x_target, q_guess, damp):
       
                count = 0
                q_sol = q_guess.copy()
                error_norm_list = []
                condition_list = []
                while True:

                        x_act = self.fk.forward_kinematics(q_sol)

                        J = self.fk.jacobian(q_sol)
                        J_H = 0.5*x_act.as_mat_right()@J
                        
                        I = np.eye(J.shape[1])
                        
                        
                       
                        
                        # error calculation
                        pos_error_norm = np.linalg.norm(x_target.getPosition() - x_act.getPosition())
                        orientation_error_norm = (x_target.real.inverse()*x_act.real).getAngle()
                        
                        error_norm = pos_error_norm + orientation_error_norm
                        
                        error_norm_list.append(error_norm)
                        condition_list.append(np.linalg.cond(J_H.T @ J_H + damp*error_norm*I))
                        
                        if error_norm < 1.0e-6:
                                return q_sol, error_norm_list, condition_list, True
                        
                        
                        J_pinv = np.linalg.inv(J_H.T @ J_H + damp*error_norm*I) @ J_H.T  

                        error = (x_target - x_act).asVector()

                        delta_q = J_pinv@error.flatten()

                        q_sol += delta_q
                        #q_sol = self.clamp(q_sol, self.min_lim, self.max_lim)

                        if count > 20:
                                break

                        count += 1

                return q_sol, error_norm_list, condition_list, False


        def getIK_classic(self, x_target, q_guess, damp):
       
                count = 0
                q_sol = q_guess.copy()
                error_norm_list = []
                condition_list = []
                while True:

                        x_act = self.fk.forward_kinematics(q_sol)

                        J = self.fk.jacobian6(q_sol)
                        
                        I = np.eye(J.shape[0])
                        
                        x_error = x_target*x_act.inverse()
                        pos_error = x_error.getPosition().flatten()
                        orientation_error = 2.0*x_error.real.log().getVector().flatten()
                        #pos_error = (x_target.getPosition() - x_act.getPosition()).flatten()
                        #orientation_error = 2*(x_target.real*x_act.real.inverse()).log().getVector().flatten()
                                     
                        # error calculation
                        pos_error_norm = np.linalg.norm(pos_error)
                        orientation_error_norm = np.linalg.norm(orientation_error)
                        #print(orientation_error_norm, np.linalg.norm(orientation_error))
                        error_norm = pos_error_norm + orientation_error_norm
                        
                        error_norm_list.append(error_norm)
                        condition_list.append(np.linalg.cond(J @ J.T + damp*error_norm*I))
                        
                        if error_norm < 1.0e-6:
                                return q_sol, error_norm_list, condition_list, True
                        
                        
                        J_pinv = J.T @ np.linalg.inv(J @ J.T + damp*error_norm*I)  

                        error = np.array([*orientation_error, *pos_error])

                        delta_q = J_pinv@error

                        q_sol += delta_q
                        #q_sol = self.clamp(q_sol, self.min_lim, self.max_lim)

                        if count > 20:
                                break

                        count += 1

                return q_sol, error_norm_list, condition_list, False
    
    
        def clamp(self, q, min_lim, max_lim):
                for i in range(6):
                        if q[i] > max_lim[i]:
                                q[i] = max_lim[i]

                        if q[i] < min_lim[i]:
                                q[i] = min_lim[i]

                return q