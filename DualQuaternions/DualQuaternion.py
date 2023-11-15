from DualQuaternions.Quaternion import Quaternion
import numpy as np


class DualQuaternion:
    def __init__(self, real, dual):
        if not isinstance(real, Quaternion) or not isinstance(dual, Quaternion):
            raise ValueError("Both real and dual parts should be instances of the Quaternion class")
        self.real = real
        self.dual = dual
        
        self.Rw = real.w
        self.Rx = real.x
        self.Ry = real.y
        self.Rz = real.z
        
        self.Dw = dual.w
        self.Dx = dual.x
        self.Dy = dual.y
        self.Dz = dual.z
        
    def __repr__(self):
        return f"DualQuaternion(Real: {self.real}, Dual: {self.dual})"

    def __add__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.real + other.real, self.dual + other.dual)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.real - other.real, self.dual - other.dual)
        return NotImplemented
    
    def __mul__(self, other):
        if isinstance(other, DualQuaternion):
            return DualQuaternion(self.real * other.real, self.real * other.dual + self.dual * other.real)
        if isinstance(other, float):
            return DualQuaternion(self.real * other, self.dual * other)
        return NotImplemented

    
    def as_mat_right(self):
        m = np.array([
            [self.Rw, -self.Rx, -self.Ry, -self.Rz, 0, 0, 0, 0],
            [self.Rx, self.Rw, self.Rz, -self.Ry, 0, 0, 0, 0],
            [self.Ry, -self.Rz, self.Rw, self.Rx, 0, 0, 0, 0],
            [self.Rz, self.Ry, -self.Rx, self.Rw, 0, 0, 0, 0],
            [self.Dw, -self.Dx, -self.Dy, -self.Dz, self.Rw, -self.Rx, -self.Ry, -self.Rz],
            [self.Dx, self.Dw, self.Dz, -self.Dy, self.Rx, self.Rw, self.Rz, -self.Ry],
            [self.Dy, -self.Dz, self.Dw, self.Dx, self.Ry, -self.Rz, self.Rw, self.Rx],
            [self.Dz, self.Dy, -self.Dx, self.Dw, self.Rz, self.Ry, -self.Rx, self.Rw]
        ])
        return m
    
    def as_mat_left(self):
        m = np.array([
            [self.Rw, -self.Rx, -self.Ry, -self.Rz, 0, 0, 0, 0],
            [self.Rx, self.Rw, -self.Rz, self.Ry, 0, 0, 0, 0],
            [self.Ry, self.Rz, self.Rw, -self.Rx, 0, 0, 0, 0],
            [self.Rz, -self.Ry, self.Rx, self.Rw, 0, 0, 0, 0],
            [self.Dw, -self.Dx, -self.Dy, -self.Dz, self.Rw, -self.Rx, -self.Ry, -self.Rz],
            [self.Dx, self.Dw, -self.Dz, self.Dy, self.Rx, self.Rw, -self.Rz, self.Ry],
            [self.Dy, self.Dz, self.Dw, -self.Dx, self.Ry, self.Rz, self.Rw, -self.Rx],
            [self.Dz, -self.Dy, self.Dx, self.Dw, self.Rz, -self.Ry, self.Rx, self.Rw]
        ])
        return m

    def asVector(self):
        vec = np.array([[self.Rw], [self.Rx], [self.Ry], [self.Rz], [self.Dw], [self.Dx], [self.Dy], [self.Dz]])
        return vec
    
    def as6Vector(self):
        vec = np.array([[self.Rx], [self.Ry], [self.Rz], [self.Dx], [self.Dy], [self.Dz]])
        return vec
    
    def first_conjugate(self):
        real_new = Quaternion(self.Rw, -self.Rx, -self.Ry, -self.Rz)
        dual_new = Quaternion(self.Dw, -self.Dx, -self.Dy, -self.Dz)
        
        return DualQuaternion(real_new, dual_new)
    
    def second_conjugate(self):
        real_new = Quaternion(self.Rw, self.Rx, self.Ry, self.Rz)
        dual_new = Quaternion(-self.Dw, -self.Dx, -self.Dy, -self.Dz)
        
        return DualQuaternion(real_new, dual_new)
    
    def third_conjugate(self):
        real_new = Quaternion(self.Rw, -self.Rx, -self.Ry, -self.Rz)
        dual_new = Quaternion(-self.Dw, self.Dx, self.Dy, self.Dz)
        
        return DualQuaternion(real_new, dual_new)
    
    def inverse(self):
        return self.first_conjugate()
    
    def asRotationDualQuaternion(self):
        pos = Quaternion(0,0,0,0)
        return DualQuaternion(self.real, pos)
    
    def power(self, t):
        return self.exp(self.log()*t)
    
    def log(self):
        ln_r = Quaternion.log(self.real)
        
        norm_complex_vector = np.linalg.norm(self.real.getVector())
        
        theta = np.arctan2(norm_complex_vector, self.Rw)
        vr = self.real.getVector()
        vd = self.dual.getVector()
        gamma = np.dot(vr.flatten(), vd.flatten())
                
        dot = np.dot(self.real.asVector().flatten(), self.dual.asVector().flatten())
        
        kr = 0
        kd = 0
        chi = 0
        if norm_complex_vector < 0.001:
            kr = (1.0 - (theta**2)/6.0 + (7.0*theta**4)/360.0)/self.real.norm()
            chi = (-2/3 - 1/5*theta**2 - 17/420*theta**4)/self.real.norm()**3
        else:
            kr = theta/np.linalg.norm(self.real.getVector())
            chi = ((self.real.w)/self.real.norm() - kr)/norm_complex_vector**2
            
        kd = gamma*chi - self.dual.w/self.real.norm()**2
        
        ln_d_vec = kd*self.real.getVector() + kr*self.dual.getVector()
        ln_d_w = dot/self.real.norm()**2
        
        ln_d_vec_flat = ln_d_vec.flatten()
        ln_d = Quaternion(ln_d_w, ln_d_vec_flat[0], ln_d_vec_flat[1], ln_d_vec_flat[2])
        
        return DualQuaternion(ln_r, ln_d)
    
    
    @classmethod
    def exp(cls, dq):
        theta = dq.real.norm()
        
        vr = dq.real.getVector()
        vd = dq.dual.getVector()
        gamma = np.dot(vr.flatten(), vd.flatten())
        
        real = Quaternion.exp(dq.real)
        
        kr = 0
        kd = 0
        if theta < 0.001:
            kr = 1 - theta**2/6 + theta**4/120
            kd = gamma*(-1/3 + theta**2/30 - theta**4/840)
        else:
            kr = np.sin(theta)/theta
            kd = gamma*(np.cos(theta) - kr)/theta**2
        
        dual_vec = (kr*dq.dual.getVector() + kd*dq.real.getVector()).flatten()
        dual = Quaternion(-kr*gamma, dual_vec[0], dual_vec[1], dual_vec[2])
        
        dual = dual*np.exp(dq.real.w) + real*dq.dual.w
        
        return cls(real, dual)
        
            
    @classmethod
    def sclerp(cls, DualQuat1, DualQuat2, s):
        rot_interp = Quaternion.slerp(DualQuat1.real, DualQuat2.real, s)
        
        pos_interp = DualQuat1.getPosition().flatten() + (DualQuat2.getPosition().flatten() - DualQuat1.getPosition().flatten())*s
        pos = Quaternion(0, pos_interp[0], pos_interp[1], pos_interp[2])
        dual = pos*rot_interp*0.5
        
        return cls(rot_interp, dual)
    
    @classmethod
    def sclerp_dot(cls, DualQuat1, DualQuat2, s, s_dot):
        r = Quaternion.slerp(DualQuat1.real, DualQuat2.real, s)
        r_dot = Quaternion.slerp_dot(DualQuat1.real, DualQuat2.real, s, s_dot)
        
        pos = DualQuat1.getPosition().flatten() + (DualQuat2.getPosition().flatten() - DualQuat1.getPosition().flatten())*s
        pos_dot = (DualQuat2.getPosition().flatten() - DualQuat1.getPosition().flatten())*s_dot
        t = Quaternion(0, pos[0], pos[1], pos[2])
        t_dot = Quaternion(0, pos_dot[0], pos_dot[1], pos_dot[2])
        
        dual1 = t_dot*r*0.5
        dual2 = t*r_dot*0.5
        
        dual_dot = dual1 + dual2
        return cls(r_dot, dual_dot)
    
    @classmethod
    def from6Vector(cls, vec):
        real = Quaternion(0, vec[0], vec[1], vec[2])
        dual = Quaternion(0, vec[3], vec[4], vec[5])
        
        return cls(real, dual)
        
    @classmethod
    def screwAxis(cls, lx, ly, lz, px, py, pz):
        vx = py*lz - pz*ly
        vy = pz*lx - px*lz
        vz = px*ly - py*lx
        
        rot = Quaternion(0, lx, ly, lz)
        dual = Quaternion(0, vx, vy, vz)
        
        return cls(rot, dual)
    
    @classmethod 
    def fromScrewAxis(cls, theta, d, a):
        rx = a.Rx
        ry = a.Ry
        rz = a.Rz
        mx = a.Dx
        my = a.Dy
        mz = a.Dz
        
        rot = Quaternion(np.cos(theta*0.5), np.sin(theta*0.5)*rx, np.sin(theta*0.5)*ry, np.sin(theta*0.5)*rz)
        
        dual = Quaternion(-d*0.5*np.sin(theta*0.5), d*0.5*rx*np.cos(theta*0.5) + mx*np.sin(theta*0.5), d*0.5*ry*np.cos(theta*0.5) + my*np.sin(theta*0.5), d*0.5*rz*np.cos(theta*0.5) + mz*np.sin(theta*0.5))
        
        return cls(rot, dual)
    
    @classmethod
    def fromTranslation(cls, t):
        rot = Quaternion(1,0,0,0)
        dual = Quaternion(0, 0.5*t[0,0], 0.5*t[1,0], 0.5*t[2,0])
        
        return cls(rot, dual)
    
    @classmethod
    def fromQuatPos(cls, real, pos):
        pos = Quaternion(0, pos[0], pos[1], pos[2])
        dual = pos*real*0.5
        
        return cls(real, dual)
    
    @classmethod
    def fromTransformation(cls, T):
        R = T[:3, :3]
        pos = T[:3, 3] 

        rot = Quaternion.fromRotation(R)
        pos = Quaternion(0, pos[0], pos[1], pos[2])
        dual = pos*rot*0.5
        return cls(rot, dual)
    
    def asTransformation(self):
        R = self.getRotationMatrix()
        pos = self.getPosition()
        
        transformation_matrix = np.identity(4)
    
        transformation_matrix[:3, :3] = R
        
        transformation_matrix[:3, 3] = pos[:, 0]
        
        return transformation_matrix
        
    @classmethod
    def fromAxisAngleTranslation(cls, theta, vec, t):
        rot = Quaternion.fromAxisAngle(theta, vec)
        pos = Quaternion(0, t[0,0], t[1,0], t[2,0])
        
        dual = pos*rot*0.5
        
        return cls(rot, dual)
    
    @classmethod
    def basicConstructor(cls, Rw, Rx, Ry, Rz, Dw, Dx, Dy, Dz):
        real = Quaternion(Rw, Rx, Ry, Rz)
        dual = Quaternion(Dw, Dx, Dy, Dz)
        
        return cls(real, dual)

    def getPosition(self):
        posQuat = self.dual*self.real.inverse()
        pos = 2*posQuat.getVector()
        return pos
    
    def getRotationMatrix(self):
        return self.real.asRotationMatrix()

