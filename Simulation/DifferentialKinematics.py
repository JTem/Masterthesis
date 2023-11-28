import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics

class DifferentialKinematics:
        def __init__(self):
                self.forward_kinematics = ForwardKinematics()
        
        
        def differential_kinematics(self, q, q_dot, Qd, Qd_dot):
                x = self.forward_kinematics.forward_kinematics(q)

                quat_real = x.real
                pos_real = x.getPosition().flatten()


                quat_des = Qd.real
                pos_des = Qd.getPosition().flatten()

                delta_pos = pos_des - pos_real

                delta_quat = quat_des*quat_real.inverse()
                o_error = delta_quat.log().getVector()
                flat_o_error = o_error.flatten()

                Omega = Qd_dot*Qd.inverse()*2.0
                x_dot = Omega.as6Vector()

                error = np.array([flat_o_error[0], flat_o_error[1], flat_o_error[2], delta_pos[0], delta_pos[1], delta_pos[2]])
                J = self.forward_kinematics.jacobian6(q)
                
                lamda = 0.0001
                I = np.eye(J.shape[1])
                J_pinv = np.linalg.inv(J.T @ J + lamda*I) @ J.T
                

                kp = 20

                vel = x_dot.flatten() + kp*error
                q_dot_ = J_pinv@vel.flatten()

                return q_dot_.flatten()
    
    


    
