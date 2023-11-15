import math
import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
     
            
    def __repr__(self):
        return f"Quaternion({self.w:.3f}, {self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)
        if isinstance(other, float):
            w = self.w*other
            x = self.x*other
            y = self.y*other
            z = self.z*other
            return Quaternion(w, x, y, z)
        return NotImplemented

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        n = self.norm()
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)
    
    def getAngle(self):
        theta = 2.0*np.arctan2(np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z), self.w)
        return theta
    
    def inverse(self):
        return self.conjugate().normalize()

    def rotate(self, v):
        v_quat = Quaternion(0, *v)
        rotated = self * v_quat * self.inverse()
        return [rotated.x, rotated.y, rotated.z]
        
        
    def log(self):
        norm_complex_vector = np.linalg.norm(self.getVector())
        
        theta_half = np.arctan2(norm_complex_vector, self.w)
        
        
        vector_scale = 0
        
        if norm_complex_vector < 0.001:
            vector_scale = (1.0 - (theta_half**2)/6.0 + (7.0*theta_half**4)/360.0)/self.norm()
        else:
            vector_scale = theta_half/norm_complex_vector
        
        vec = (self.getVector()*vector_scale).flatten()
        return Quaternion(np.log(np.linalg.norm(self.asVector())), vec[0], vec[1], vec[2])
    
    @classmethod
    def exp(cls, q):
        theta_half = np.linalg.norm(q.getVector())
        
        frac_sin_theta_half = 0
        
        if theta_half < 0.001:
            frac_sin_theta_half = 1 - theta_half**2/6.0 + theta_half**4/120.0
        else:
            frac_sin_theta_half = np.sin(theta_half)/theta_half
            
        scaled_rot_axis = (q.getVector()*frac_sin_theta_half).flatten()
        
        return cls(np.cos(theta_half), scaled_rot_axis[0], scaled_rot_axis[1], scaled_rot_axis[2])*np.exp(q.w)

    def exp_dot(q):
        theta_half = np.linalg.norm(q.getVector())
        
        x = q.x
        y = q.y
        z = q.z
        
        frac_sin_theta_half = 0
        frac_cos_sin = 0
        
        if theta_half < 0.001:
            frac_sin_theta_half = 1 - theta_half**2/6.0 + theta_half**4/120.0
            frac_cos_sin = -1/3
        else:
            frac_sin_theta_half = np.sin(theta_half)/theta_half
            frac_cos_sin = np.cos(theta_half)/theta_half**2 - np.sin(theta_half)/theta_half**3
            
        dexp = np.zeros((4,3))
        
        dfw_dx = -frac_sin_theta_half*x
        dfx_dx = frac_cos_sin*x*x + frac_sin_theta_half
        dfy_dx = frac_cos_sin*x*y
        dfz_dx = frac_cos_sin*x*z
        
        dfw_dy = -frac_sin_theta_half*y
        dfx_dy = frac_cos_sin*y*x
        dfy_dy = frac_cos_sin*y*y + frac_sin_theta_half
        dfz_dy = frac_cos_sin*x*z
        
        dfw_dz = -frac_sin_theta_half*z
        dfx_dz = frac_cos_sin*z*x
        dfy_dz = frac_cos_sin*z*y
        dfz_dz = frac_cos_sin*z*z + frac_sin_theta_half
        
        dexp[0,0] = dfw_dx
        dexp[1,0] = dfx_dx
        dexp[2,0] = dfy_dx
        dexp[3,0] = dfz_dx
        
        dexp[0,1] = dfw_dy
        dexp[1,1] = dfx_dy
        dexp[2,1] = dfy_dy
        dexp[3,1] = dfz_dy
        
        dexp[0,2] = dfw_dz
        dexp[1,2] = dfx_dz
        dexp[2,2] = dfy_dz
        dexp[3,2] = dfz_dz
        
        return dexp
        
    def exp_ddot(q):
        theta_half = np.linalg.norm(q.getVector())
        
        x = q.x
        y = q.y
        z = q.z
        
        frac_sin_theta_half = 0
        frac_cos_sin = 0
        frac_cos_sin_sin = 0
        
        if theta_half < 0.001:
                frac_sin_theta_half = 1 - theta_half**2/6.0 + theta_half**4/120.0
                frac_cos_sin = -1/3
                frac_cos_sin_sin = 0.0666667
        else:
                frac_sin_theta_half = np.sin(theta_half)/theta_half
                frac_cos_sin = np.cos(theta_half)/theta_half**2 - np.sin(theta_half)/theta_half**3
                frac_cos_sin_sin =-3*np.cos(theta_half)/theta_half**4 - np.sin(theta_half)/theta_half**3 + 3*np.sin(theta_half)/theta_half**5
            
        ddexp = np.zeros((4,3,3))
        
        # note that the 2nd derivatives are symmetric, thus dxdy = dydx
        ddfw_dxdx = -frac_cos_sin*x*x - frac_sin_theta_half
        ddfw_dydx = -frac_cos_sin*x*y 
        ddfw_dzdx = -frac_cos_sin*x*z
        
        ddfw_dydy = -frac_cos_sin*y*y - frac_sin_theta_half
        ddfw_dzdy = -frac_cos_sin*y*z
        
        ddfw_dzdz = -frac_cos_sin*z*z - frac_sin_theta_half
        
        #derivative of x by tangent space vector elements x y z
        ddfxdxdx = x*x*x*frac_cos_sin_sin + 3*x*frac_cos_sin
        ddfxdydx = x*x*y*frac_cos_sin_sin +   y*frac_cos_sin
        ddfxdzdx = x*x*z*frac_cos_sin_sin +   z*frac_cos_sin

        ddfxdydy = x*y*y*frac_cos_sin_sin + x*frac_cos_sin
        ddfxdzdy = x*y*z*frac_cos_sin_sin
        
        ddfxdzdz = x*z*z*frac_cos_sin_sin + x*frac_cos_sin
        
        #derivative of y by tangent space vector elements x y z
        ddfydxdx = x*x*y*frac_cos_sin_sin + y*frac_cos_sin
        ddfydydx = x*y*y*frac_cos_sin_sin + x*frac_cos_sin
        ddfydzdx = x*y*z*frac_cos_sin_sin

        ddfydydy = y*y*y*frac_cos_sin_sin + 3*y*frac_cos_sin
        ddfydzdy = y*y*z*frac_cos_sin_sin + z*frac_cos_sin
        
        ddfydzdz = y*z*z*frac_cos_sin_sin + y*frac_cos_sin
        
        #derivative of y by tangent space vector elements x y z
        ddfzdxdx = x*x*z*frac_cos_sin_sin + z*frac_cos_sin
        ddfzdydx = x*y*z*frac_cos_sin_sin 
        ddfzdzdx = x*z*z*frac_cos_sin_sin + x*frac_cos_sin

        ddfzdydy = y*y*z*frac_cos_sin_sin + z*frac_cos_sin
        ddfzdzdy = y*z*z*frac_cos_sin_sin + y*frac_cos_sin
        
        ddfzdzdz = z*z*z*frac_cos_sin_sin + 3*z*frac_cos_sin
        
        # assign values to tensor
        ddexp[0,0,0] = ddfwdxdx
        ddexp[1,0,0] = ddfxdxdx
        ddexp[2,0,0] = ddfydxdx
        ddexp[3,0,0] = ddfzdxdx

        ddexp[0,1,0] = ddfwdydx
        ddexp[1,1,0] = ddfxdydx
        ddexp[2,1,0] = ddfydydx
        ddexp[3,1,0] = ddfzdydx
        
        ddexp[0,2,0] = ddfwdzdx
        ddexp[1,2,0] = ddfxdzdx
        ddexp[2,2,0] = ddfydzdx
        ddexp[3,2,0] = ddfzdzdx
        
        
        ddexp[0,0,1] = ddfwdydx
        ddexp[1,0,1] = ddfxdydx
        ddexp[2,0,1] = ddfydydx
        ddexp[3,0,1] = ddfzdydx
        
        ddexp[0,1,1] = ddfwdydy
        ddexp[1,1,1] = ddfxdydy
        ddexp[2,1,1] = ddfydydy
        ddexp[3,1,1] = ddfzdydy
        
        ddexp[0,2,1] = ddfwdzdy
        ddexp[1,2,1] = ddfxdzdy
        ddexp[2,2,1] = ddfydzdy
        ddexp[3,2,1] = ddfzdzdy
        
        
        ddexp[0,0,2] = ddfwdzdx
        ddexp[1,0,2] = ddfxdzdx
        ddexp[2,0,2] = ddfydzdx
        ddexp[3,0,2] = ddfzdzdx
        
        ddexp[0,1,2] = ddfwdzdy
        ddexp[1,1,2] = ddfxdzdy
        ddexp[2,1,2] = ddfydzdy
        ddexp[3,1,2] = ddfzdzdy
        
        ddexp[0,2,2] = ddfwdzdz
        ddexp[1,2,2] = ddfxdzdz
        ddexp[2,2,2] = ddfydzdz
        ddexp[3,2,2] = ddfzdzdz
        
        return ddexp
        
    @classmethod
    def fromRotation(cls, R):
        r11 = R[0,0]
        r21 = R[0,1]
        r31 = R[0,2]
        
        r12 = R[1,0]
        r22 = R[1,1]
        r32 = R[1,2]
        
        r13 = R[2,0]
        r23 = R[2,1]
        r33 = R[2,2]
        t = 0.0
        
        if r33 < 0: 
            if r11 > r22:
                t = 1.0 + r11 - r22 - r33;
                x = 0.5/np.sqrt(t)*t
                y = 0.5/np.sqrt(t)*(r12 + r21)
                z = 0.5/np.sqrt(t)*(r31 + r13)
                w = 0.5/np.sqrt(t)*(r23 - r32)
                
            else:
                t = 1.0 - r11 + r22 - r33;
                x = 0.5/np.sqrt(t)*(r12+r21)
                y = 0.5/np.sqrt(t)*t
                z = 0.5/np.sqrt(t)*(r23 + r32)
                w = 0.5/np.sqrt(t)*(r31 - r13)
                
        else:
            if r11 < -r22:
                t = 1.0 - r11 - r22 + r33
                
                x = 0.5/np.sqrt(t)*(r31 + r13)
                y = 0.5/np.sqrt(t)*(r23 + r32)
                z = 0.5/np.sqrt(t)*t
                w = 0.5/np.sqrt(t)*(r12 - r21)
            else:
                t = 1.0 + r11 + r22 + r33
                
                x = 0.5/np.sqrt(t)*(r23 - r32)
                y = 0.5/np.sqrt(t)*(r31 - r13)
                z = 0.5/np.sqrt(t)*(r12 - r21)
                w = 0.5/np.sqrt(t)*t
                
        return cls(w, x, y, z)


    @classmethod
    def slerp(cls, qa, qb, s):
        qm = Quaternion(1,0,0,0)
        
        dot_product = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z

        short_path = False
        if short_path and dot_product < 0.0:
            qb = Quaternion(-qb.w, -qb.x, -qb.y, -qb.z)
            dot_product = -dot_product

        
        halfTheta = math.atan2(math.sqrt(1.0 - dot_product*dot_product), dot_product)
        
        # If the angle is close to 2*pi, return the starting quaternion to avoid the singularity
        if abs(halfTheta - math.pi) < 1e-6:
            return qa
            
        sinHalfTheta = math.sin(halfTheta)
   
        ratioA = math.sin((1.0-s)*halfTheta)/sinHalfTheta
        ratioB = math.sin(s*halfTheta)/sinHalfTheta
        
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.x = (qa.x * ratioA + qb.x * ratioB)
        qm.y = (qa.y * ratioA + qb.y * ratioB)
        qm.z = (qa.z * ratioA + qb.z * ratioB)
        
        #qm.normalize()
        return qm

    @classmethod
    def slerp_old(cls, qa, qb, s):
        qm = Quaternion(1,0,0,0)
        
        ratioA = 1.0 - s
        ratioB = s
        
        cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z
        abs_cosHalfTheta = abs(cosHalfTheta)
        
        if abs_cosHalfTheta < 0.999:
            halfTheta = math.acos(cosHalfTheta)
            sinHalfTheta = math.sin(halfTheta)
            ratioA = math.sin((1.0-s)*halfTheta)/sinHalfTheta
            ratioB = math.sin(s*halfTheta)/sinHalfTheta
        
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.x = (qa.x * ratioA + qb.x * ratioB)
        qm.y = (qa.y * ratioA + qb.y * ratioB)
        qm.z = (qa.z * ratioA + qb.z * ratioB)
		
        return qm

    @classmethod
    def slerp_dot(cls, qa, qb, s, s_dot):
        qm = Quaternion(0,0,0,0)
        
        dot_product = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z

        short_path = False
        if short_path and dot_product < 0.0:
            qb = Quaternion(-qb.w, -qb.x, -qb.y, -qb.z)
            dot_product = -dot_product

        
        halfTheta = math.atan2(math.sqrt(1.0 - dot_product*dot_product), dot_product)

        # If the angle is close to 2*pi, return the starting quaternion to avoid the singularity
        if abs(halfTheta - math.pi) < 1e-6:
            return Quaternion(0,0,0,0)
            
        sinHalfTheta = math.sin(halfTheta)
        ratioA = (-math.cos((1 - s) * halfTheta) * halfTheta * s_dot) / sinHalfTheta
        ratioB = (math.cos(s * halfTheta) * halfTheta * s_dot) / sinHalfTheta
        
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.x = (qa.x * ratioA + qb.x * ratioB)
        qm.y = (qa.y * ratioA + qb.y * ratioB)
        qm.z = (qa.z * ratioA + qb.z * ratioB)

        return qm

    @classmethod
    def slerp_ddot(cls, qa, qb, s, s_dot, s_ddot):
        qm = Quaternion(0,0,0,0)
        
        dot_product = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z

        short_path = False
        if short_path and dot_product < 0.0:
            qb = Quaternion(-qb.w, -qb.x, -qb.y, -qb.z)
            dot_product = -dot_product

        halfTheta = math.atan2(math.sqrt(1.0 - dot_product*dot_product), dot_product)

        # If the angle is close to 2*pi, return the starting quaternion to avoid the singularity
        if abs(halfTheta - math.pi) < 1e-6:
            return Quaternion(0,0,0,0)
            
        sinHalfTheta = math.sin(halfTheta)
        ratioA = -(halfTheta * math.cos(halfTheta * (1 - s)) * s_ddot
                + halfTheta * halfTheta * math.sin(halfTheta * (1 - s)) * s_dot * s_dot) / sinHalfTheta;
        ratioB = (halfTheta * math.cos(halfTheta * s) * s_ddot
                - (halfTheta * halfTheta * math.sin(halfTheta * s) * s_dot * s_dot)) / sinHalfTheta;
        
        
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.x = (qa.x * ratioA + qb.x * ratioB)
        qm.y = (qa.y * ratioA + qb.y * ratioB)
        qm.z = (qa.z * ratioA + qb.z * ratioB)
        
        return qm

    
    @classmethod
    def slerp_dot_old(cls, qa, qb, s, s_dot):
        qm = Quaternion(1,0,0,0)
        
        ratioA = 1.0 - s
        ratioB = s
        
        cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z
        abs_cosHalfTheta = abs(cosHalfTheta)
        
        if abs_cosHalfTheta < 0.999:
            halfTheta = math.acos(abs_cosHalfTheta)
            sinHalfTheta = math.sin(halfTheta)
            ratioA = (-math.cos((1 - s) * halfTheta) * halfTheta * s_dot) / sinHalfTheta
            ratioB = (math.cos(s * halfTheta) * halfTheta * s_dot) / sinHalfTheta
        
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.x = (qa.x * ratioA + qb.x * ratioB)
        qm.y = (qa.y * ratioA + qb.y * ratioB)
        qm.z = (qa.z * ratioA + qb.z * ratioB)
		
        return qm
    
    @classmethod
    def slerp_ddot_old(cls, qa, qb, s, s_dot, s_ddot):
        qm = Quaternion(1,0,0,0)
        
        ratioA = 1.0 - s
        ratioB = s
        
        cosHalfTheta = qa.w * qb.w + qa.x * qb.x + qa.y * qb.y + qa.z * qb.z
        abs_cosHalfTheta = abs(cosHalfTheta)
        
        if abs_cosHalfTheta < 0.999:
            halfTheta = math.acos(abs_cosHalfTheta)
            sinHalfTheta = math.sin(halfTheta)
            ratioA = -(halfTheta * math.cos(halfTheta * (1 - s)) * s_ddot
					+ halfTheta * halfTheta * math.sin(halfTheta * (1 - s)) * s_dot * s_dot) / sinHalfTheta;
            ratioB = (halfTheta * math.cos(halfTheta * s) * s_ddot
					- (halfTheta * halfTheta * math.sin(halfTheta * s) * s_dot * s_dot)) / sinHalfTheta;
        
        qm.w = (qa.w * ratioA + qb.w * ratioB)
        qm.x = (qa.x * ratioA + qb.x * ratioB)
        qm.y = (qa.y * ratioA + qb.y * ratioB)
        qm.z = (qa.z * ratioA + qb.z * ratioB)
		
        return qm
    
    # @classmethod
    # def fromAxisAngle(cls, theta, vec):
    #     w = np.cos(theta*0.5)
    #     v = vec*np.sin(theta*0.5);
    #     return cls(w, v[0,0], v[1,0], v[2,0])
    
    @classmethod
    def fromAxisAngle(cls, theta, vec):
        w = np.cos(theta*0.5)
        v = vec*np.sin(theta*0.5);
        return cls(w, v[0], v[1], v[2])
    
    
    def asVector(self):
        vec = np.array([[self.w], [self.x], [self.y], [self.z]])
        return vec
    
    def getVector(self):
        vec = np.array([[self.x], [self.y], [self.z]])
        return vec
    
    def asRotationMatrix(self):
        w, x, y, z = self.w, self.x, self.y, self.z
        R = np.array([
            [2*(w*w + x*x) - 1, 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 2*(w*w + y*y) - 1, 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 2*(w*w + z*z)-1]])

        return R
