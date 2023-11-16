import math
import numpy as np
from DualQuaternionQuinticBlends.Arc import Arc


class ArcGenerator:
        def __init__(self, DQ_list, velocity, angular_velocity_max):
                self.DQ = DQ_list
                self.velocity = velocity
                self.angular_velocity_max = angular_velocity_max
                self.number_of_arcs = len(DQ_list)-2
                self.arc_list = []
                is_last_arc = False
                for i in range(self.number_of_arcs):

                        if i == self.number_of_arcs - 1:
                                is_last_arc = True

                        self.generateArcs(DQ_list[i], DQ_list[i+1], DQ_list[i+2], is_last_arc)
        
        
        def calcCenterAndRadius(self, p1, p2, p3):
                v = p3-p2
                u = p3-p1
                t = p2-p1

                w = np.cross(t,u)

                center = p1 + (u*np.dot(t,t) * np.dot(u,v) - t*np.dot(u,u) * np.dot(t,v))*0.5/np.linalg.norm(w)**2

                radius = np.linalg.norm((p2 - center))

                return center, radius
        
        def computeRotationMatrix(v_center_start, v_center_mid):
                z_ = np.cross(v_start_center, v_mid_center)
                z = z_/np.linalg.norm(z_)

                y = np.cross(z, v_center_start)

                y = y_/np.linalg.norm(y_)
                
                rotation = np.array([v_center_start, y,  z]).transpose()
                
                return rotation
                
        def generateArcs(self, dq1, dq2, dq3, is_last_arc):
                
                p1 = dq1.getPosition()
                p2 = dq2.getPosition()
                p3 = dq3.getPosition()
                
                center, radius = self.calcCenterAndRadius(p1, p2, p3)
                
                v_center_start_ = p1 - center
                v_center_mid_ = p2 - center
                v_center_end_ = p3 - center
                
                v_center_start = v_center_start_/np.linalg.norm(v_center_start_)
                v_center_mid = v_center_mid_/np.linalg.norm(v_center_mid_)
                v_center_end = v_center_end_/np.linalg.norm(v_center_end_)

                rotation = computeRotationMatrix(v_center_start, v_center_mid)
  
                angle = self.calculateAngleBetweenVectors(v_start_center, v_mid_center)
                        
                self.arc_list.append(Arc(dq1, dq2, self.velocity, self.angular_velocity_max, center, radius, rotation, angle))
                
                if is_last_arc:
                                
                        rotation = computeRotationMatrix(v_center_start, v_center_mid)
                        angle = self.calculateAngleBetweenVectors(v_mid_center, v_end_center)
                        
                        self.arc_list.append(Arc(dq2, dq3, self.velocity, self.angular_velocity_max, center, radius, rotation, angle))
        
        
        def calculateAngleBetweenVectors(self, v1, v2):
                return math.atan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))

        
        def getArcList(self):
                return self.arc_list
    
            
    
