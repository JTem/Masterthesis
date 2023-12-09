import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion

class ForwardKinematics:
        def __init__(self, type):

                self.d1 = 0.438
                self.d3 = 0.7
                self.d5 = 0.7
                self.d7 = 0.115
                self.PEE = np.array([0, 0, 1.953 + 0.3599])
                tool_orientation = Quaternion.fromAxisAngle(np.pi*0.25, np.array([0,1,0]))
              
                self.M = DualQuaternion.fromQuatPos(tool_orientation, self.PEE)

                self.s1 = DualQuaternion.screwAxis(0,0,1, 0,0,0)
                self.s2 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1)
                self.s3 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1)
                self.s4 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1+self.d3)
                self.s5 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1+self.d3)
                self.s6 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1+self.d3+self.d5)
                self.s7 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1+self.d3+self.d5)
                

                self.screws_0 = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7]
                self.dof = 7
                
                if type == "extended":
                        tool_rotation_axis = tool_orientation*Quaternion(0,0,0,1)*tool_orientation.inverse()
                        self.s8 = DualQuaternion.screwAxis(*tool_rotation_axis.getVector().flatten(), 0,0,self.d1+self.d3+self.d5 + 0.3599)
                        self.screw_0.append(self.s8)
                
                        self.dof = 8
    
        def forward_kinematics(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                for i in range(self.dof):
                        x = x*DualQuaternion.exp(0.5*theta[i]*self.screws_0[i])

                return x*self.M
    
        def jacobian8(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((8, self.dof))

                # loop over all joints
                for i in range(self.dof):

                        # get screw (defined in base frame) from list
                        s_0 = self.screws_0[i]

                        #transform screw with the line transformation
                        s_i = x*s_0*x.inverse()

                        # progress transformation with screw and joint angle
                        x = x*DualQuaternion.exp(0.5*theta[i]*s_0)

                        J[:, i] = s_i.asVector().flatten()

                return J

#         def jacobian_body(self, theta):
#                 x = self.forward_kinematics(theta)
                
#                 J = self.jacobian8(theta)
                
#                 Jb = x.inverse().as_mat_left()@x.as_mat_right()@J
                
#                 return np.vstack([Jb[1:4], Jb[5:8]])
        
        
        def jacobian(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((6, len(theta)))

                # loop over all joints
                for i in range(len(theta)):

                        # get screw (defined in base frame) from list
                        s_0 = self.screws_0[i]

                        #transform screw with the line transformation
                        s_i = x*s_0*x.inverse()

                        # progress transformation with screw and joint angle
                        x = x*DualQuaternion.exp(0.5*theta[i]*s_0)

                        J[:, i] = s_i.as6Vector().flatten()

                return J
        
        
        def jacobian_dot(self, theta, theta_dot):

                # initialize transformation as identity transformation
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)

                # initialialize transformation derivative as zero
                x_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)

                # initialize Jacobian derivative with zeros
                J_dot = np.zeros((8, len(theta)))

                # loop over all joints
                for i in range(len(theta)):
                        # get screw (defined in base frame) from list
                        s_0 = self.screws_0[i]

                        #compute screw derivative and transform
                        s_i_dot = x_dot*s_0*x.inverse() + x*s_0*x_dot.inverse()

                        # compute temporary transformation via exponential map
                        x_temp = DualQuaternion.exp(0.5*theta[i]*s_0)

                        # compute temporary transformation derivative via exponential derivative
                        x_dot_temp = DualQuaternion.exp(0.5*theta[i]*s_0)*(0.5*theta_dot[i]*s_0)

                        # update transformation derivative
                        x_dot = x_dot*x_temp + x*x_dot_temp

                        # update transformation
                        x = x*x_temp

                        #assign screw derivative to jacobian derivative row
                        J_dot[:, i] = s_i_dot.asVector().flatten()

                return J_dot
        

        def hessian(self, theta):
                h = 0.000001
                J = self.jacobian(theta)

                # Initialize the Hessian tensor
                H = np.zeros((6, self.dof, self.dof))

                for i in range(n):
                        theta_temp = theta.copy()
                        theta_temp[i] += h
                        J_temp = self.jacobian(theta_temp)
                        H[:, :, i] = (J_temp - J) / h

                return H
