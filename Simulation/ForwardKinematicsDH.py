import numpy as np

class ForwardKinematicsDH:
        def __init__(self, fk_type = "normal"):
                
                self.dof = 7
                
                self.a1 = 0
                self.a2 = 0
                self.a3 = 0
                self.a4 = 0
                self.a5 = 0
                self.a6 = 0
                self.a7 = 0

                self.d1 = 0.438
                self.d2 = 0
                self.d3 = 0.7
                self.d4 = 0
                self.d5 = 0.7
                self.d6 = 0
                self.d7 = 0.115

                self.alpha1 = -np.pi*0.5
                self.alpha2 = np.pi*0.5
                self.alpha3 = -np.pi*0.5
                self.alpha4 = np.pi*0.5
                self.alpha5 = -np.pi*0.5
                self.alpha6 = np.pi*0.5
                self.alpha7 = 0.0

                self.theta1 = 0
                self.theta2 = 0
                self.theta3 = 0
                self.theta4 = 0
                self.theta5 = 0
                self.theta6 = 0
                self.theta7 = 0

        def dh_transform(self, d, a, alpha, theta):
                # Create transformation matrix
                T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                          [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                          [0,              np.sin(alpha),               np.cos(alpha),               d],
                          [0,              0,                           0,                           1]])
                return T


        def getFK(self, q):
                T01 = self.dh_transform(self.d1, self.a1, self.alpha1, self.theta1 + q[0])
                T12 = self.dh_transform(self.d2, self.a2, self.alpha2, self.theta2 + q[1])
                T23 = self.dh_transform(self.d3, self.a3, self.alpha3, self.theta3 + q[2])
                T34 = self.dh_transform(self.d4, self.a4, self.alpha4, self.theta4 + q[3])
                T45 = self.dh_transform(self.d5, self.a5, self.alpha5, self.theta5 + q[4])
                T56 = self.dh_transform(self.d6, self.a6, self.alpha6, self.theta6 + q[5])
                T67 = self.dh_transform(self.d7, self.a7, self.alpha7, self.theta7 + q[6])


                T07 = T01@T12@T23@T34@T45@T56@T67

                return T07

        def getFK_full(self, q):
                T00 = np.eye(4)
                T01 = self.dh_transform(self.d1, self.a1, self.alpha1, self.theta1 + q[0])
                T12 = self.dh_transform(self.d2, self.a2, self.alpha2, self.theta2 + q[1])
                T23 = self.dh_transform(self.d3, self.a3, self.alpha3, self.theta3 + q[2])
                T34 = self.dh_transform(self.d4, self.a4, self.alpha4, self.theta4 + q[3])
                T45 = self.dh_transform(self.d5, self.a5, self.alpha5, self.theta5 + q[4])
                T56 = self.dh_transform(self.d6, self.a6, self.alpha6, self.theta6 + q[5])
                T67 = self.dh_transform(self.d7, self.a7, self.alpha7, self.theta7 + q[6])
                
                T0 = [0]*8
                
                T0[0] = T00
                T0[1] = T01
                T0[2] = T0[1]@T12
                T0[3] = T0[2]@T23
                T0[4] = T0[3]@T34
                T0[5] = T0[4]@T45
                T0[6] = T0[5]@T56
                T0[7] = T0[6]@T67
        
                return T0
        
        
        def cross(self, a, b):
                r = np.array([[a[1][0]*b[2][0] - a[2][0]*b[1][0]], [a[2][0]*b[0][0] - a[0][0]*b[2][0]], [a[0][0]*b[1][0] - a[1][0]*b[0][0]]])
                
                return r
        
        def getSpaceJacobian(self, q):
                
                T0 = self.getFK_full(q)
                
                J = np.zeros((6,7))
                
                for i in range(self.dof):
                        joint_axis = T0[i][0:3, 2:3]
                        pos = T0[i][0:3, 3:4]
                        
                        J[0:3, i:i+1] = joint_axis
                        J[3:6, i:i+1] = self.cross(pos, joint_axis)
                
                return J
                        