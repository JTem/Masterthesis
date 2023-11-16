import numpy as np
import matplotlib.pyplot as plt
from DualQuaternions import Quaternion

class PoseTrajGen:
    def __init__(self, Segments, a_max, j_max, ang_acc_max, ang_jerk_max):
        self.Segments = Segments
       
        self.a_max = a_max if a_max >= 0.1 else 0.1
        self.j_max = j_max if j_max >= 1 else 1
        
        self.ang_acc_max = ang_acc_max if ang_acc_max >= 0.1 else 0.1
        self.ang_jerk_max = ang_jerk_max if ang_jerk_max >= 1 else 1
        
        self.min_blend_duration = 0.002
        
        for s in self.Segments:
            if s.duration < self.min_blend_duration:
                self.Segments.remove(s)
        
        self.num_segments = len(Segments)
        
        self.T_blend_list_cart = []
        self.T_blend_list_quat = []
        
        self.p0_list = []
        self.p1_list = []
        self.v0_list = []
        self.v1_list = []
        self.a0_list = []
        self.a1_list = []
        
        self.ang_v0_list = []
        self.ang_v1_list = []
        
        self.time_blend_start_cart = []
        self.time_blend_start_quat = []
        
        self.time_vector = []
        self.generate_traj_params()
    
    def spline5(self, q0, q1, v0, v1, a0, a1, t, T):
        h = q1 - q0
        
        p0 = q0
        p1 = v0
        p2 = 0.5*a0
        p3 = 1/(2*T**3)*(20*h - (8*v1 + 12*v0)*T - (3*a0 - a1)*T**2)
        p4 = 1/(2*T**4)*(-30*h + (14*v1 + 16*v0)*T + (3*a0 - 2*a1)*T**2)
        p5 = 1/(2*T**5)*(12*h - 6*(v1 + v0)*T + (a1-a0)*T**2)
        
        
        pos = p0 + p1*t + p2*t**2 + p3*t**3 +p4*t**4 + p5*t**5
        vel = p1 + 2*p2*t + 3*p3*t**2 + 4*p4*t**3 + 5*p5*t**4
        acc = 2*p2 + 6*p3*t + 12*p4*t**2 + 20*p5*t**3
        jerk = 6*p3 + 24*p4*t + 60*p5*t**2
        #snap = 24*p4 + 120*p5*t
        
        return pos, vel, acc, jerk
    
        
    def spline3(self, v0, v1, a0, a1, t, T):
        p0 = v0
        p1 = a0
        p2 = (3.0*(v1 - v0) - (2.0*a0 + a1)*T)/T**2
        p3 = (-2.0*(v1 - v0) + (a0 + a1)*T)/T**3
    
        q = p0 + p1*t + p2*t**2 + p3*t**3
        q_dot = p1 + 2*p2*t + 3*p3*t**2
        q_ddot = 2*p2 + 6*p3*t
        q_dddot = 6*p3
        return q, q_dot, q_ddot, q_dddot
        
    def spline3_pos(self, v0, v1, a0, a1, t, T):
        p0 = v0
        p1 = a0
        p2 = (3.0*(v1 - v0) - (2.0*a0 + a1)*T)/T**2
        p3 = (-2.0*(v1 - v0) + (a0 + a1)*T)/T**3
    
        pos = p0*t + 0.5*p1*t**2 + 1.0/3.0*p2*t**3 + 0.25*p3*t**4
        q = p0 + p1*t + p2*t**2 + p3*t**3
        q_dot = p1 + 2*p2*t + 3*p3*t**2
        q_ddot = 2*p2 + 6*p3*t

        return pos, q, q_dot, q_ddot
    
    
    def jerk_gradient(self, q0, q1, v0, v1, a0, a1, t, T):
        h = q1 - q0
        
        p3 = 1/(2*T**3)*(20*h - (8*v1 + 12*v0)*T - (3*a0 - a1)*T**2)
        p4 = 1/(2*T**4)*(-30*h + (14*v1 + 16*v0)*T + (3*a0 - 2*a1)*T**2)
        p5 = 1/(2*T**5)*(12*h - 6*(v1 + v0)*T + (a1-a0)*T**2)
        
        eps = 1e-6
        t1 = t + eps
        t2 = t - eps
        
        jerk1 = np.linalg.norm(6*p3 + 24*p4*t1 + 60*p5*t1**2)
        jerk2 = np.linalg.norm(6*p3 + 24*p4*t2 + 60*p5*t2**2)
        gradient = (jerk1 - jerk2)/(2*eps)
        
        return gradient
        
    def max_acc_jerk_p5(self, q0, q1, v0, v1, a0, a1, T):
        
        t = self.calc_time(v0, v1, a0, a1, T)
        print("cubic time: ", t)
        print("total_time: ", T)
        for i in range(10):
            gradient = self.jerk_gradient(q0, q1, v0, v1, a0, a1, t, T)
            t -= 0.05/(i+1)*T*np.sign(gradient)
        print("quintic time: ", t)
        
        
        _, _, acc1, jerk1 = self.spline5(q0, q1,v0,v1,a0,a1, 0, T)
        _, _, acc2, jerk2 = self.spline5(q0, q1,v0,v1,a0,a1, T, T)
        _, _, acc3, _ = self.spline5(q0, q1,v0,v1,a0,a1, t, T)
        
        print("acc1: ", np.linalg.norm(acc1))
        print("acc2: ", np.linalg.norm(acc2))
        print("acc3: ", np.linalg.norm(acc3))
        
        jerk_max = max([np.linalg.norm(jerk1), np.linalg.norm(jerk2)])
        acc_max = max([np.linalg.norm(acc1), np.linalg.norm(acc2), np.linalg.norm(acc3)])
    
        return acc_max, jerk_max
        
    
    def calc_time(self, v0, v1, a0, a1, T):
        p2 = (3*(v1 - v0) - (2*a0 + a1)*T)/T**2
        p3 = (-2*(v1 - v0) + (a0 + a1)*T)/T**3
        
        t = 0
        if np.linalg.norm(p3) > 1e-6:
            t = -np.dot(p2, p3)/(3*np.linalg.norm(p3)**2)
        
        return t

    def max_acc_jerk_p3(self, v0, v1, a0, a1, T):
        t = self.calc_time(v0, v1, a0, a1, T)
        
        _, acc1, jerk1, _ = self.spline3(v0,v1,a0,a1, 0, T)
        _, acc2, jerk2, _ = self.spline3(v0,v1,a0,a1, T, T)
        _, acc3, _, _ = self.spline3(v0,v1,a0,a1, t, T)
        
        jerk_max = max([np.linalg.norm(jerk1), np.linalg.norm(jerk2)])
        acc_max = max([np.linalg.norm(acc1), np.linalg.norm(acc2), np.linalg.norm(acc3)])
       
        return acc_max, jerk_max


    def generate_traj_params(self):    
        
        # define blending times (for now only set 0.1)
        self.T_blend_list_cart.append(0)
        self.T_blend_list_quat.append(0)
        for i in range(0, self.num_segments-1):
            self.T_blend_list_cart.append(self.min_blend_duration)
            self.T_blend_list_quat.append(self.min_blend_duration)
        self.T_blend_list_cart.append(0)
        self.T_blend_list_quat.append(0)
        
        
        for i in range(0, len(self.T_blend_list_cart)):
            
            if self.T_blend_list_cart[i] > 0:
                half_T_blend = self.T_blend_list_cart[i]*0.5
                first_segment = self.Segments[i-1]
                second_segment = self.Segments[i]
                first_segment_duration = first_segment.duration
                
                
                self.p0_list.append(first_segment.getPos(first_segment_duration - half_T_blend))
                self.p1_list.append(second_segment.getPos(half_T_blend))
                
                self.ang_v0_list.append(first_segment.getAngularVel())
                self.ang_v1_list.append(second_segment.getAngularVel())
                
                self.v0_list.append(first_segment.getVel(first_segment_duration - half_T_blend))
                self.v1_list.append(second_segment.getVel(half_T_blend))
                
                self.a0_list.append(first_segment.getAcc(first_segment_duration - half_T_blend))
                self.a1_list.append(second_segment.getAcc(half_T_blend))
            else:
                self.p0_list.append(np.array([0,0,0]))
                self.p1_list.append(np.array([0,0,0]))
                
                self.ang_v0_list.append(np.array([0,0,0]))
                self.ang_v1_list.append(np.array([0,0,0]))
                
                self.v0_list.append(np.array([0,0,0]))
                self.v1_list.append(np.array([0,0,0]))
                self.a0_list.append(np.array([0,0,0]))
                self.a1_list.append(np.array([0,0,0]))
                
        
        # print(self.v0_list)
        # print(self.v1_list)
        # print(self.ang_v0_list)
        # print(self.ang_v1_list)
        
 
        blend_phase_overlap_cart = True
        max_acc_limit_violated_cart = True
        
        blend_phase_overlap_quat = True
        max_acc_limit_violated_quat = True
        
        cnt = 0
        while (blend_phase_overlap_cart or blend_phase_overlap_quat or max_acc_limit_violated_cart or max_acc_limit_violated_quat) or cnt < 10:
              cnt += 1
      
              print("cnt: ", cnt)
              blend_phase_overlap_cart = False
              blend_phase_overlap_quat = False
              for i in range(0, self.num_segments-1):
                  
                  if 0.5*(self.T_blend_list_cart[i] + self.T_blend_list_cart[i+1]) > self.Segments[i].duration:
                          
                      print("blendphases cart overlapped")
                    
                      self.Segments[i].duration *= 1.05    
                      
                      blend_phase_overlap_cart = True
                      
                      
                      seg = self.Segments[i]
                      self.p1_list[i] = (seg.getPos(self.T_blend_list_cart[i]*0.5))
                      self.v1_list[i] = (seg.getVel(self.T_blend_list_cart[i]*0.5))
                      self.a1_list[i] = (seg.getAcc(self.T_blend_list_cart[i]*0.5))
                      
                      self.p0_list[i+1] = (seg.getPos(seg.duration - self.T_blend_list_cart[i+1]*0.5))
                      self.v0_list[i+1] = (seg.getVel(seg.duration - self.T_blend_list_cart[i+1]*0.5))
                      self.a0_list[i+1] = (seg.getAcc(seg.duration - self.T_blend_list_cart[i+1]*0.5))
                      
                      self.ang_v1_list[i] = (seg.getAngularVel())
                      self.ang_v0_list[i+1] = (seg.getAngularVel())
             
                      
                  if 0.5*(self.T_blend_list_quat[i] + self.T_blend_list_quat[i+1]) > self.Segments[i].duration:
                        
                      print("blendphases quat overlapped")
                      
                      self.Segments[i].duration *= 1.05  
                      
                      blend_phase_overlap_quat = True
                      
                      seg = self.Segments[i]
                      self.p1_list[i] = (seg.getPos(self.T_blend_list_cart[i]*0.5))
                      self.v1_list[i] = (seg.getVel(self.T_blend_list_cart[i]*0.5))
                      self.a1_list[i] = (seg.getAcc(self.T_blend_list_cart[i]*0.5))
                      
                      self.p0_list[i+1] = (seg.getPos(seg.duration - self.T_blend_list_cart[i+1]*0.5))
                      self.v0_list[i+1] = (seg.getVel(seg.duration - self.T_blend_list_cart[i+1]*0.5))
                      self.a0_list[i+1] = (seg.getAcc(seg.duration - self.T_blend_list_cart[i+1]*0.5))
                      
                      self.ang_v1_list[i] = (seg.getAngularVel())
                      self.ang_v0_list[i+1] = (seg.getAngularVel())
                       
                     
                    
              max_acc_limit_violated_cart = False  
              max_acc_limit_violated_quat = False 
              for i in range(0,len(self.T_blend_list_cart)):
                   if self.T_blend_list_cart[i] > 0:
                  

                       max_acc, max_jerk = self.max_acc_jerk_p5(self.p0_list[i], self.p1_list[i], self.v0_list[i], self.v1_list[i], self.a0_list[i], self.a1_list[i], self.T_blend_list_cart[i]) 
                     
                       a_lim = max([self.a_max, np.linalg.norm(self.a0_list[i]), np.linalg.norm(self.a1_list[i])])
                       lamda_acc = max_acc/a_lim
                       
                       print("lamda_acc", lamda_acc)
                       if max_acc > a_lim*1.01:
                           print("acceleration limit violated")
                           max_acc_limit_violated_cart = True
                       print("duration_blend_list_before", self.T_blend_list_cart[i])
                       self.T_blend_list_cart[i] *= lamda_acc
                       #print("duration_blend_list_after", self.T_blend_list_cart[i])
                       if self.T_blend_list_cart[i] < self.min_blend_duration:
                           self.T_blend_list_cart[i] = self.min_blend_duration
                      
                       seg1 = self.Segments[i-1]
                       seg2 = self.Segments[i]
                       half_T_blend = self.T_blend_list_cart[i]*0.5
                       
                       self.p0_list[i] = (seg1.getPos(seg1.duration - half_T_blend))
                       self.p1_list[i] = (seg2.getPos(half_T_blend))
                       
                       self.v0_list[i] = (seg1.getVel(seg1.duration - half_T_blend))
                       self.v1_list[i] = (seg2.getVel(half_T_blend))
                       
                       self.a0_list[i] = (seg1.getAcc(seg1.duration - half_T_blend))
                       self.a1_list[i] = (seg2.getAcc(half_T_blend))
                    
                       
                       max_acc, max_jerk = self.max_acc_jerk_p5(self.p0_list[i], self.p1_list[i], self.v0_list[i], self.v1_list[i], self.a0_list[i], self.a1_list[i], self.T_blend_list_cart[i]) 
                       #print("max_jerk: ", max_jerk)
                    
                       if max_jerk > self.j_max:
                           #print("jerk limit violated")
                           self.T_blend_list_cart[i] *= np.sqrt(max_jerk/self.j_max)
                           
                           if self.T_blend_list_cart[i] < self.min_blend_duration:
                               self.T_blend_list_cart[i] = self.min_blend_duration
                           
                           half_T_blend = self.T_blend_list_cart[i]*0.5

                           self.p0_list[i] = (seg1.getPos(seg1.duration - half_T_blend))
                           self.p1_list[i] = (seg2.getPos(half_T_blend))

                           self.v0_list[i] = (seg1.getVel(seg1.duration - half_T_blend))
                           self.v1_list[i] = (seg2.getVel(half_T_blend))

                           self.a0_list[i] = (seg1.getAcc(seg1.duration - half_T_blend))
                           self.a1_list[i] = (seg2.getAcc(half_T_blend))
                       
                       
                       
                       
                   if self.T_blend_list_quat[i] > 0:
                   
                       max_angular_acc, max_angular_jerk = self.max_acc_jerk_p3(self.ang_v0_list[i], self.ang_v1_list[i], np.array([0,0,0]), np.array([0,0,0]), self.T_blend_list_quat[i]) 
                      
                       lamda_acc = max_angular_acc/self.ang_acc_max
                        
                       #print("max ang acc: ", max_angular_acc)
                       if max_angular_acc > self.ang_acc_max*1.01:
                           
                           print("angular acceleration limit violated")
                           max_acc_limit_violated_quat = True
                            
                           self.T_blend_list_quat[i] *= lamda_acc
                        
                           if self.T_blend_list_quat[i] < self.min_blend_duration:
                               self.T_blend_list_quat[i] = self.min_blend_duration
    
                           
                           self.ang_v0_list[i] = self.Segments[i-1].getAngularVel()
                           self.ang_v1_list[i] = self.Segments[i].getAngularVel()
                       
                    
                       max_angular_acc, max_angular_jerk = self.max_acc_jerk_p3(self.ang_v0_list[i], self.ang_v1_list[i], np.array([0,0,0]), np.array([0,0,0]), self.T_blend_list_quat[i]) 
                       
                       #print("max jerk acc: ", max_angular_jerk)
                       
                       if max_angular_jerk > self.ang_jerk_max:
                           self.T_blend_list_quat[i] *= np.sqrt(max_angular_jerk/self.ang_jerk_max)
                           
                           if self.T_blend_list_quat[i] < self.min_blend_duration:
                               self.T_blend_list_quat[i] = self.min_blend_duration
                            
                       self.ang_v0_list[i] = self.Segments[i-1].getAngularVel()
                       self.ang_v1_list[i] = self.Segments[i].getAngularVel()
                         
                        
        self.time_blend_start_cart.append(0)
        self.time_blend_start_quat.append(0)
        self.time_vector.append(0)
            
        for i in range(0,self.num_segments):
            duration_sum = 0
            for j in range(0, i+1):
                duration_sum += self.Segments[j].duration
            self.time_vector.append(duration_sum)
            self.time_blend_start_cart.append(duration_sum - self.T_blend_list_cart[i+1]*0.5)              
            self.time_blend_start_quat.append(duration_sum - self.T_blend_list_quat[i+1]*0.5)      
                    
        
        
    def determine_segment_type(self, t, time_blend_start, T_blend_list):
        # first find in which area we are
        cnt = 0
        for i in range(0, len(time_blend_start)):
            if t < time_blend_start[i]:
                cnt = i-1
                break;
        
        if t >= time_blend_start[-1]:
            cnt =  len(time_blend_start)-2
       
        if t > time_blend_start[cnt] and t <= time_blend_start[cnt] + T_blend_list[cnt]:
            return "blend", cnt
    
        else:
            return "lin", cnt
        
    def online_interpolation(self, t):
        seg_cart, cnt_cart = self.determine_segment_type(t, self.time_blend_start_cart, self.T_blend_list_cart)
        
        seg_quat, cnt_quat = self.determine_segment_type(t, self.time_blend_start_quat, self.T_blend_list_quat)
        
        if seg_cart == "blend":
            dt = t - self.time_blend_start_cart[cnt_cart]
            pos, vel, acc, jerk = self.spline5(self.p0_list[cnt_cart], self.p1_list[cnt_cart], self.v0_list[cnt_cart], self.v1_list[cnt_cart], self.a0_list[cnt_cart], self.a1_list[cnt_cart], dt, self.T_blend_list_cart[cnt_cart])

        if seg_cart == "lin":
            dt = t - self.time_blend_start_cart[cnt_cart] - 0.5*self.T_blend_list_cart[cnt_cart]
            jerk = np.array([0,0,0])
            acc = self.Segments[cnt_cart].getAcc(dt)
            vel = self.Segments[cnt_cart].getVel(dt)
            pos = self.Segments[cnt_cart].getPos(dt)
            
            
        if seg_quat == "blend":
            dt = t - self.time_blend_start_quat[cnt_quat]
            log, w, w_dot, w_ddot = self.spline3_pos(self.ang_v0_list[cnt_quat], self.ang_v1_list[cnt_quat], np.array([0,0,0]), np.array([0,0,0]), dt, self.T_blend_list_quat[cnt_quat])
            
            
            log -= self.ang_v0_list[cnt_quat]*0.5*self.T_blend_list_quat[cnt_quat]
            
            quaternion = Quaternion.exp(Quaternion(0,log[0], log[1], log[2])*0.5)*self.Segments[cnt_quat].q1
            
        if seg_quat == "lin":
            dt = t - self.time_blend_start_quat[cnt_quat] - 0.5*self.T_blend_list_quat[cnt_quat]
            w_ddot = np.array([0,0,0])
            w_dot = np.array([0,0,0])
            w = self.Segments[cnt_quat].getAngularVel()
            
            log = w*dt
            
            quaternion = Quaternion.exp(Quaternion(0, log[0], log[1], log[2])*0.5)*self.Segments[cnt_quat].q1
            #pos = self.Segments[cnt_cart].getPos(dt)
        
        return pos, vel, acc, jerk, quaternion, w, w_dot, w_ddot
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    