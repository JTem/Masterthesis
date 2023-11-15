import math
import numpy as np
from DualQuaternionBlends.PoseArc import Arc


class PoseC:
    def __init__(self, P, Q, velocity, angular_velocity_max):
        self.P = P
        self.velocity = velocity
        self.angular_velocity_max = angular_velocity_max
        self.number_of_arcs = len(P)-2
        self.arc_list = []
        is_last_arc = False
        for i in range(self.number_of_arcs):
            
            if i == self.number_of_arcs - 1:
                is_last_arc = True
            
            self.generateArcs(P[i], P[i+1], P[i+2], Q[i], Q[i+1], Q[i+2], is_last_arc)
        
       # self.generateLines()
    
    
    def calcCenterAndRadius(self, p1, p2, p3):
        v = p3-p2
        u = p3-p1
        t = p2-p1
        
        w = np.cross(t,u)
        
        center = p1 + (u*np.dot(t,t) * np.dot(u,v) - t*np.dot(u,u) * np.dot(t,v))*0.5/np.linalg.norm(w)**2
        
        radius = np.linalg.norm((p2 - center))
        
        return center, radius
    
    def generateArcs(self, p1, p2, p3, q1, q2, q3, is_last_arc):

        
     
        center, radius = self.calcCenterAndRadius(p1, p2, p3)
        
        v_start_center_ = p1 - center
        v_mid_center_ = p2 - center
        v_end_center_ = p3 - center
        
        v_start_center = v_start_center_/np.linalg.norm(v_start_center_)
        v_mid_center = v_mid_center_/np.linalg.norm(v_mid_center_)
        v_end_center = v_end_center_/np.linalg.norm(v_end_center_)
        
        z_ = np.cross(v_start_center, v_mid_center)
        z = z_/np.linalg.norm(z_)
        
        y_ = np.cross(z, v_start_center)
        
        y = y_/np.linalg.norm(y_)
        
        rotation = np.array([v_start_center, y,  z]).transpose()
        print(rotation)
        angle1 = self.calculateAngleBetweenVectors(v_start_center, v_mid_center)
        angle2 = self.calculateAngleBetweenVectors(v_mid_center, v_end_center)
        
        if is_last_arc:
            self.arc_list.append(Arc(p1, q1, q3, self.velocity, self.angular_velocity_max, center, radius, rotation, angle1 + angle2))
        else:
            self.arc_list.append(Arc(p1, q1, q2, self.velocity, self.angular_velocity_max, center, radius, rotation, angle1))
            
        
        
    def calculateAngleBetweenVectors(self, v1, v2):
        return math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

    def getArcList(self):
        return self.arc_list
    
            
    
