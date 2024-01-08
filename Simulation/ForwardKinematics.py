import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion

class ForwardKinematics:
        def __init__(self, fk_type = "normal"):
                
                # Define the offset distances (d_i) for each joint
                self.d1 = 0.438
                self.d3 = 0.7
                self.d5 = 0.7
                self.d7 = 0.115
                
                # Define the end-effector position and orientation based on fk_type
                self.PEE = np.array([0, 0, 1.953])
                tool_orientation = Quaternion(1,0,0,0)
                
                # Adjust the end-effector parameters for different fk_type (e.g., "weld" or "extended")
                if fk_type == "weld" or fk_type == "extended":
                        self.PEE = np.array([0, 0, 1.953 + 0.3599])
                        tool_orientation = Quaternion.fromAxisAngle(np.pi*0.25, np.array([0,1,0]))
                
                # Calculate the home configuration dual quaternion
                self.M = DualQuaternion.fromQuatPos(tool_orientation, self.PEE)
                
                # Define the screw axes for each joint in space (s) and body (b) frames
                self.s1 = DualQuaternion.screwAxis(0,0,1, 0,0,0)
                self.s2 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1)
                self.s3 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1)
                self.s4 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1+self.d3)
                self.s5 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1+self.d3)
                self.s6 = DualQuaternion.screwAxis(0,1,0, 0,0,self.d1+self.d3+self.d5)
                self.s7 = DualQuaternion.screwAxis(0,0,1, 0,0,self.d1+self.d3+self.d5)
                
                # Convert screw axes to body frame using the transformation M
                self.b1 = self.M.inverse()*self.s1*self.M
                self.b2 = self.M.inverse()*self.s2*self.M
                self.b3 = self.M.inverse()*self.s3*self.M
                self.b4 = self.M.inverse()*self.s4*self.M
                self.b5 = self.M.inverse()*self.s5*self.M
                self.b6 = self.M.inverse()*self.s6*self.M
                self.b7 = self.M.inverse()*self.s7*self.M
                
                # Create lists of all screw axes in space and body frames
                self.screws_s = [self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7]
                self.screws_b = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7]
                
                self.dof = 7
                
                # Additional screw axis and dof adjustment for extended configuration
                if fk_type == "extended":
                        tool_rotation_axis = tool_orientation*Quaternion(0,0,0,1)*tool_orientation.inverse()
                        self.s8 = DualQuaternion.screwAxis(*tool_rotation_axis.getVector().flatten(), 0,0,self.d1+self.d3+self.d5+self.d7 + 0.3599)
                        self.b8 = self.M.inverse()*self.s8*self.M
                        self.screws_s.append(self.s8)
                        self.screws_b.append(self.b8)
                
                        self.dof = 8
        
        # Method to calculate the forward kinematics using space frame screws
        def getFK(self, theta):
                return self.getSpaceFK(theta)
        
        
        def getSpaceFK(self, theta):
                # Initialize the identity dual quaternion
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                
                # Multiply the exponentials of the screw axes to calculate the end-effector pose
                for i in range(self.dof):
                        x = x*DualQuaternion.exp(0.5*theta[i]*self.screws_s[i])
                
                # Apply the home configuration transformation M
                return x*self.M
            
                
        def getBodyFK(self, theta):
                # Similar to getSpaceFK but using body frame screws
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                for i in range(self.dof):
                        x = x*DualQuaternion.exp(0.5*theta[i]*self.screws_b[i])
                
                # Apply the home configuration transformation M
                return self.M*x
        
        
        # Various methods to calculate Jacobians and Hessian follow similar structure
        # They involve calculating the appropriate screw axis transformation and
        # progressively transforming it based on joint angles
        
        # compute 8-dimensional Jacobian in space frame
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
        
        # compute 8-dimensional Jacobian in body frame
        def getBodyJacobian8(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((8, self.dof))
                
                for i in range(self.dof-1,-1, -1):
                        s_body = self.screws_b[i]
                    
                        s_i = x*s_body*x.inverse()
                    
                        x = x*DualQuaternion.exp(-0.5*theta[i]*s_body)
                    
                        J[:, i] = s_i.asVector().flatten()
                    
                return J
        
        
        # compute 6-dimensional Jacobian in space frame
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
        
        # compute 6-dimensional Jacobian in body frame
        def getBodyJacobian(self, theta):
            
                x = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                J = np.zeros((6, self.dof))
                
                for i in range(self.dof-1,-1, -1):
                        s_body = self.screws_b[i]
                    
                        s_i = x*s_body*x.inverse()
                    
                        x = x*DualQuaternion.exp(-0.5*theta[i]*s_body)
                    
                        J[:, i] = s_i.as6Vector().flatten()
                    
                return J
            
        # compute 6-dimensional geometric Jacobian (rotated body Jacobian)
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
            
        
        # Hessian calculation for second-order partial derivatives
        def getHessian(self, theta):
                h = 0.000001
                
                # Calculate the body Jacobian at the given joint angles theta
                J = self.getBodyJacobian(theta)

                # Initialize the Hessian tensor
                H = np.zeros((6, self.dof, self.dof))
                
                # Iterate over all degrees of freedom to calculate partial derivatives of Jacobian entries
                for i in range(self.dof):
                        # Copy the original joint angles to a temporary variable
                        theta_temp = theta.copy()
                        
                        # Increment the ith joint angle by a small amount h
                        theta_temp[i] += h
                        
                        # Calculate the body Jacobian at the slightly altered joint angles theta_temp
                        J_temp = self.getBodyJacobian(theta_temp)
                        
                         
                        # Calculate the partial derivative of Jacobian with respect to the ith joint angle
                        # This is done using numerical differentiation (central difference method)
                        H[:, :, i] = (J_temp - J) / h
                        
                # Return the Hessian tensor
                return H
