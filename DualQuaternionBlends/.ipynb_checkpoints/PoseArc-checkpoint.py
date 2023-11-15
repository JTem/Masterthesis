import math
import numpy as np
from DualQuaternions.Quaternion import Quaternion

class Arc:
    def __init__(self, p, q1, q2, velocity, angular_velocity_max, center, radius, rotation, angle):
        self.p = p
        self.q1 = q1
        self.q2 = q2
        self.velocity = velocity
        self.center = center
        self.radius = radius
        self.rotation = rotation
        self.angle = angle
        self.angular_velocity_max = angular_velocity_max
        
        self.dist = angle * radius
        
        self.duration = self.dist/self.velocity
        
        dq = q1.inverse()*q2
        self.theta = abs(dq.getAngle())
        e = dq.log()
        if e.norm() > 1e-6:
            e = e.normalize()
            
        e0 = q1*e*q1.inverse()
        self.rotation_axis = e0.getVector().flatten()
        
        if self.duration > 1e-6:
            self.angular_velocity = self.theta/self.duration
            
            if self.angular_velocity > self.angular_velocity_max:
                self.duration = self.theta / self.angular_velocity_max
                self.angular_velocity = self.angular_velocity_max
                
        else: 
            self.duration = self.theta / self.angular_velocity_max
            self.angular_velocity = self.angular_velocity_max
            
            
            

        
        
    def getPos(self, time):
        point_on_arc = np.array([self.radius*np.cos(time/self.duration*self.angle), self.radius*np.sin(time/self.duration*self.angle), 0])
        
        translation = self.center + self.rotation@point_on_arc
        
        return translation
    
    
    def getAngularVel(self):
        return self.rotation_axis*self.theta/self.duration
    
    
    def getVel(self, time):
         # Derivatives of the position function
         dx_dt = -self.radius * np.sin(time/self.duration * self.angle) * self.angle/self.duration
         dy_dt = self.radius * np.cos(time/self.duration * self.angle) * self.angle/self.duration
         dz_dt = 0
    
         velocity = np.array([dx_dt, dy_dt, dz_dt])
         rotated_velocity = self.rotation @ velocity
         return rotated_velocity

    def getAcc(self, time):
        # Second derivatives of the position function
        d2x_dt2 = -self.radius * np.cos(time/self.duration * self.angle) * (self.angle/self.duration)**2
        d2y_dt2 = -self.radius * np.sin(time/self.duration * self.angle) * (self.angle/self.duration)**2
        d2z_dt2 = 0
           
        acceleration = np.array([d2x_dt2, d2y_dt2, d2z_dt2])
        rotated_acceleration = self.rotation @ acceleration
        return rotated_acceleration