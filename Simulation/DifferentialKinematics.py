import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics

class DifferentialKinematics:
    
        def __init__(self):
            
                self.fk = ForwardKinematics()
                self.dof = self.fk.dof

        
        def differential_kinematics(self, q, q_dot, DQd, DQd_dot):
                x = self.forward_kinematics.forward_kinematics(q)
                
                x_error = DQd*x.inverse()
                pos_error = x_error.getPosition()
                o_error = x_error.real.log().getVector()

                Omega = DQd_dot*DQd.inverse()*2.0
                x_dot = Omega.as6Vector()

                error = np.array([*o_error.flatten(), *pos_error.flatten()])
                J = self.forward_kinematics.jacobian6(q)
                
                kp = 20
                vel = x_dot.flatten() + kp*error
                pinv = np.linalg.pinv(J)
                self.gradient = self.dir_manipulability_gradient2(q)
                q_dot_ = pinv@vel.flatten() + 5.0*(np.eye(self.dof)-pinv@J)@self.gradient
                
                return q_dot_.flatten()
        
    
        def differential_kinematics_DQ(self, q, q_dot, DQd, DQd_dot):
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                
                pinv = np.linalg.pinv(J_H)
                
                kp = 20

                vel = DQd_dot.asVector() + kp*error
                q_dot_ = pinv@vel.flatten()

                return q_dot_.flatten()
        
        
               