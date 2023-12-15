import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion

class ForwardKinematics:
        def __init__(self, fk_type = "normal"):

                self.d1 = 0.438
                self.d3 = 0.7
                self.d5 = 0.7
                self.d7 = 0.115
                
                
                self.PEE = np.array([0, 0, 1.953])
                tool_orientation = Quaternion(1,0,0,0)
                
                if fk_type == "weld" or fk_type == "extended":
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
                
                self.b1 = self.M.inverse()*self.s1*self.M
                self.b2 = self.M.inverse()*self.s2*self.M
                self.b3 = self.M.inverse()*self.s3*self.M
                self.b4 = self.M.inverse()*self.s4*self.M
                self.b5 = self.M.inverse()*self.s5*self.M
                self.b6 = self.M.inverse()*self.s6*self.M
                self.b7 = self.M.inverse()*self.s7*self.M

                self.screws_s = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7]
                self.screws_b = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7]
                
                self.dof = 7
                
                if fk_type == "extended":
                        tool_rotation_axis = tool_orientation*Quaternion(0,0,0,1)*tool_orientation.inverse()
                        self.s8 = DualQuaternion.screwAxis(*tool_rotation_axis.getVector().flatten(), 0,0,self.d1+self.d3+self.d5+self.d7 + 0.3599)
                        self.b8 = self.M.inverse()*self.s8*self.M
                        self.screws_s.append(self.s8)
                        self.screws_b.append(self.b8)
                
                        self.dof = 8
        
        def getFK(self, theta):
                return self.getSpaceFK(theta)
        
        def getSpaceFK(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                for i in range(self.dof):
                        x = x*DualQuaternion.exp(0.5*theta[i]*self.screws_s[i])

                return x*self.M
            
        def getBodyFK(self, theta):
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                for i in range(self.dof):
                        x = x*DualQuaternion.exp(0.5*theta[i]*self.screws_b[i])

                return self.M*x
            
        def getSpaceJacobian8(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((8, self.dof))

                # loop over all joints
                for i in range(self.dof):

                        # get screw (defined in base frame) from list
                        s_space = self.screws_s[i]

                        #transform screw with the line transformation
                        s_i = x*s_space*x.inverse()

                        # progress transformation with screw and joint angle
                        x = x*DualQuaternion.exp(0.5*theta[i]*s_space)

                        J[:, i] = s_i.asVector().flatten()

                return J

        def getBodyJacobian8(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((8, self.dof))
                
                for i in range(self.dof-1,-1, -1):
                    s_body = self.screws_b[i]
                    
                    s_i = x*s_body*x.inverse()
                    
                    x = x*DualQuaternion.exp(-0.5*theta[i]*s_body)
                    
                    J[:, i] = s_i.asVector().flatten()
                    
                return J
        
        
        def getSpaceJacobian(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((6, self.dof))

                # loop over all joints
                for i in range(self.dof):

                        # get screw (defined in base frame) from list
                        s_space = self.screws_s[i]

                        #transform screw with the line transformation
                        s_i = x*s_space*x.inverse()

                        # progress transformation with screw and joint angle
                        x = x*DualQuaternion.exp(0.5*theta[i]*s_space)

                        J[:, i] = s_i.as6Vector().flatten()

                return J

        def getBodyJacobian(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((6, self.dof))
                
                for i in range(self.dof-1,-1, -1):
                        s_body = self.screws_b[i]
                    
                        s_i = x*s_body*x.inverse()
                    
                        x = x*DualQuaternion.exp(-0.5*theta[i]*s_body)
                    
                        J[:, i] = s_i.as6Vector().flatten()
                    
                return J
            
        def getGeometricJacobian(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                rot = self.getFK(theta).real
                rotation = DualQuaternion(rot, Quaternion(0,0,0,0))
                J = np.zeros((6, self.dof))
                
                for i in range(self.dof-1,-1, -1):
                    s_body = self.screws_b[i]
                    
                    s_i = rotation*x*s_body*x.inverse()*rotation.inverse()
                    
                    x = x*DualQuaternion.exp(-0.5*theta[i]*s_body)
                    
                    J[:, i] = s_i.as6Vector().flatten()
                    
                return J
            

        def getHessian(self, theta):
                h = 0.000001
                J = self.getBodyJacobian(theta)

                # Initialize the Hessian tensor
                H = np.zeros((6, self.dof, self.dof))

                for i in range(self.dof):
                        theta_temp = theta.copy()
                        theta_temp[i] += h
                        J_temp = self.getBodyJacobian(theta_temp)
                        H[:, :, i] = (J_temp - J) / h

                return H
