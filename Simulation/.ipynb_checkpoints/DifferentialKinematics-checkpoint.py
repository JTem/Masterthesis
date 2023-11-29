import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics

class DifferentialKinematics:
        def __init__(self):
                self.forward_kinematics = ForwardKinematics()
        
        
        def differential_kinematics(self, q, q_dot, DQd, DQd_dot):
                x = self.forward_kinematics.forward_kinematics(q)

                quat_real = x.real
                pos_real = x.getPosition().flatten()


                quat_des = DQd.real
                pos_des = DQd.getPosition().flatten()

                delta_pos = pos_des - pos_real

                delta_quat = quat_des*quat_real.inverse()
                o_error = delta_quat.log().getVector()
                flat_o_error = o_error.flatten()

                Omega = DQd_dot*DQd.inverse()*2.0
                x_dot = Omega.as6Vector()

                error = np.array([flat_o_error[0], flat_o_error[1], flat_o_error[2], delta_pos[0], delta_pos[1], delta_pos[2]])
                J = self.forward_kinematics.jacobian6(q)
                
                lamda = 0.0001
                I = np.eye(J.shape[1])
                J_pinv = np.linalg.inv(J.T @ J + lamda*I) @ J.T
                

                kp = 10

                vel = x_dot.flatten() + kp*error
                q_dot_ = J_pinv@vel.flatten()

                return q_dot_.flatten()
        
    
        def differential_kinematics_DQ(self, q, q_dot, DQd, DQd_dot):
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                
                lamda = 0.0001
                I = np.eye(J_H.shape[1])
                J_pinv = np.linalg.inv(J_H.T @ J_H + lamda*I) @ J_H.T
                

                kp = 0

                vel = DQd_dot.asVector() + kp*error
                q_dot_ = J_pinv@vel.flatten()

                return q_dot_.flatten()