import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion

class ForwardKinematics:
        def __init__(self):

                self.d1 = 0.438
                self.d3 = 0.7
                self.d5 = 0.7
                self.d7 = 0.115
                self.PEE = np.array([0, 0, 1.953 ])#+ 0.3599])
                tool_orientation = Quaternion.fromAxisAngle(np.pi*0.0, np.array([0,1,0]))
                self.M = DualQuaternion.fromQuatPos(tool_orientation, self.PEE)

                self.s1 = DualQuaternion.screwAxis(0,0,1, 0,0,0)
                self.s2 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1)
                self.s3 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1)
                self.s4 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1+self.d3)
                self.s5 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1+self.d3)
                self.s6 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1+self.d3+self.d5)
                self.s7 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1+self.d3+self.d5)

                self.screws_0 = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7]
    
        def forward_kinematics(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                for i in range(len(theta)):
                        x = x*DualQuaternion.exp(0.5*theta[i]*self.screws_0[i])

                return x*self.M
    
        def jacobian(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((8, len(theta)))

                # loop over all joints
                for i in range(len(theta)):

                        # get screw (defined in base frame) from list
                        s_0 = self.screws_0[i]

                        #transform screw with the line transformation
                        s_i = x*s_0*x.inverse()

                        # progress transformation with screw and joint angle
                        x = x*DualQuaternion.exp(0.5*theta[i]*s_0)

                        J[:, i] = s_i.asVector().flatten()

                return J

        def jacobian_body(self, theta):
                x = self.forward_kinematics(theta)
                
                J = self.jacobian(theta)
                
                Jb = x.inverse().as_mat_left()@x.as_mat_right()@J
                
                #print(Jb)
                

                return np.vstack([Jb[1:4], Jb[5:8]])
        
        def jacobian6(self, theta):
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
        
        def jacobian_dot6(self, theta, theta_dot):

                # initialize transformation as identity transformation
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)

                # initialialize transformation derivative as zero
                x_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)

                # initialize Jacobian derivative with zeros
                J_dot = np.zeros((6, len(theta)))

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
                        J_dot[:, i] = s_i_dot.as6Vector().flatten()

                return J_dot
        
        import numpy as np

        def hessian(self, theta):
                h = 0.000001
                j = self.jacobian_body(theta)

                # Assuming jacobian() returns a numpy array
                n = j.shape[1]  # Number of joints

                # Initialize the Hessian tensor
                H = np.zeros((j.shape[0], j.shape[1], n))

                for i in range(n):
                        theta_temp = theta.copy()
                        theta_temp[i] += h
                        j_temp = self.jacobian_body(theta_temp)
                        H[:, :, i] = (j_temp - j) / h

                # Multiply the Hessian tensor with theta_dot
                #result = np.tensordot(H, theta_dot, axes=([2], [0]))

                return H


    
        def hessian0(self, theta):
                
                def cross(a, b):
                        x = a[1] * b[2] - a[2] * b[1]
                        y = a[2] * b[0] - a[0] * b[2]
                        z = a[0] * b[1] - a[1] * b[0]
                        return np.array([x, y, z])
                
                n = 7
                J = self.jacobian6(theta)
                
                H = np.zeros((n, 6, n))
                for j in range(n):
                        for i in range(j, n):
                                H[j, :3, i] = cross(J[3:, j], J[:3, i])
                                H[j, 3:, i] = cross(J[3:, j], J[3:, i])

                                if i != j:
                                        H[i, :3, j] = H[j, :3, i]

                return H