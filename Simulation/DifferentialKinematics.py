import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.Manipulability import Manipulability

class DifferentialKinematics:
    
        def __init__(self, fk_type = "classic"):
                
                self.fk_type = fk_type
                self.fk = ForwardKinematics(self.fk_type)
                self.mp = Manipulability(fk_type)
                self.dof = self.fk.dof

        
        def differential_kinematics(self, q, q_dot, DQd, DQd_dot):
                x = self.fk.getFK(q)
                
                x_error = DQd*x.inverse()
                pos_error = x_error.getPosition()
                o_error = x_error.real.log().getVector()

                Omega = DQd_dot*DQd.inverse()*2.0
                x_dot = Omega.as6Vector()

                error = np.array([*o_error.flatten(), *pos_error.flatten()])
                J = self.fk.getSpaceJacobian(q)
                
                kp = 20
                vel = x_dot.flatten() + kp*error
                pinv = np.linalg.pinv(J)
                self.gradient = self.dir_manipulability_gradient2(q)
                q_dot_ = pinv@vel.flatten() + 5.0*(np.eye(self.dof)-pinv@J)@self.gradient
                
                return q_dot_.flatten(), 1
        
    
        def differential_kinematics_DQ(self, q, q_dot, DQd, DQd_dot):
                x = self.fk.getFK(q)
                
                error = (DQd - x).asVector()
                
                J = self.fk.getSpaceJacobian8(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                
                pinv = np.linalg.pinv(J_H)
                
                kp = 20

                vel = DQd_dot.asVector() + kp*error
                self.gradient = self.mp.manipulability_gradient(q)
                q_dot_ = pinv@vel.flatten() + 5.0*(np.eye(self.dof)-pinv@J_H)@self.gradient

                return q_dot_.flatten(), 1
        
        
               