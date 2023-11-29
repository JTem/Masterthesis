import numpy as np
import matplotlib.pyplot as plt
from neura_dual_quaternions import DualQuaternion, Quaternion

class DQQBTrajectoryGenerator:
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

                self.duration_blend_list_cart = []
                self.duration_blend_list_quat = []

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
                #self.generateTrajectoryParameters()
                self.generateTrajectoryParametersRadius(0.01)
    
        def quinticPolynomial(self, pos0, pos1, vel0, vel1, acc0, acc1, t, T):
                h = pos1 - pos0

                p0 = pos0
                p1 = vel0
                p2 = 0.5*acc0
                p3 = 1/(2*T**3)*(20*h - (8*vel1 + 12*vel0)*T - (3*acc0 - acc1)*T**2)
                p4 = 1/(2*T**4)*(-30*h + (14*vel1 + 16*vel0)*T + (3*acc0 - 2*acc1)*T**2)
                p5 = 1/(2*T**5)*(12*h - 6*(vel1 + vel0)*T + (acc1 - acc0)*T**2)


                pos = p0 + p1*t + p2*t**2 + p3*t**3 +p4*t**4 + p5*t**5
                vel = p1 + 2*p2*t + 3*p3*t**2 + 4*p4*t**3 + 5*p5*t**4
                acc = 2*p2 + 6*p3*t + 12*p4*t**2 + 20*p5*t**3
                jerk = 6*p3 + 24*p4*t + 60*p5*t**2

                return pos, vel, acc, jerk
    
        
        def cubicPolynomial(self, vel0, vel1, acc0, acc1, t, T):
                h = vel1 - vel0
                
                p0 = vel0
                p1 = acc0
                p2 = (3.0*h - (2.0*acc0 + acc1)*T)/T**2
                p3 = (-2.0*h + (acc0 + acc1)*T)/T**3
                
                pos = p0*t + 0.5*p1*t**2 + 1.0/3.0*p2*t**3 + 0.25*p3*t**4
                vel = p0 + p1*t + p2*t**2 + p3*t**3
                acc = p1 + 2*p2*t + 3*p3*t**2
                jerk = 2*p2 + 6*p3*t
                
                return pos, vel, acc, jerk
        
        def calcMinJerkTimeCubic(self, vel0, vel1, acc0, acc1, T):
                h = vel1 - vel0
                
                p2 = (3.0*h - (2.0*acc0 + acc1)*T)/T**2
                p3 = (-2.0*h + (acc0 + acc1)*T)/T**3

                time = 0
                if np.linalg.norm(p3) > 1e-6:
                        time = -np.dot(p2, p3)/(3*np.linalg.norm(p3)**2)

                return time
        
        def maxAccJerkQuintic(self, pos0, pos1, vel0, vel1, acc0, acc1, T):
        
                t = self.calcMinJerkTimeCubic(vel0, vel1, acc0, acc1, T)

                for i in range(10):
                        gradient = self.quinticJerkGradient(pos0, pos1, vel0, vel1, acc0, acc1, t, T)
                        t -= 0.05/(i+1)*T*np.sign(gradient)

                _, _, acceleration1, jerk1 = self.quinticPolynomial(pos0, pos1, vel0, vel1, acc0, acc1, 0, T)
                _, _, acceleration2, jerk2 = self.quinticPolynomial(pos0, pos1, vel0, vel1, acc0, acc1, T, T)
                _, _, acceleration3, _ = self.quinticPolynomial(pos0, pos1, vel0, vel1, acc0, acc1, t, T)

                jerk_max = max([np.linalg.norm(jerk1), np.linalg.norm(jerk2)])
                acc_max = max([np.linalg.norm(acceleration1), np.linalg.norm(acceleration2), np.linalg.norm(acceleration3)])

                return acc_max, jerk_max
        
        def quinticJerkGradient(self, pos0, pos1, vel0, vel1, acc0, acc1, t, T):
                eps = 1e-6
                t1 = t + eps
                t2 = t - eps

                _, _, _, jerk1 = self.quinticPolynomial(pos0, pos1, vel0, vel1, acc0, acc1, t1, T)
                _, _, _, jerk2 = self.quinticPolynomial(pos0, pos1, vel0, vel1, acc0, acc1, t2, T)
                gradient = (jerk1 - jerk2)/(2*eps)

                return gradient
        
        def maxAccJerkCubic(self, vel0, vel1, acc0, acc1, T):
                t = self.calcMinJerkTimeCubic(vel0, vel1, acc0, acc1, T)

                _, _, acceleration1, jerk1 = self.cubicPolynomial(vel0, vel1, acc0, acc1, 0, T)
                _, _, acceleration2, jerk2 = self.cubicPolynomial(vel0, vel1, acc0, acc1, T, T)
                _, _, acceleration3, _ = self.cubicPolynomial(vel0, vel1, acc0, acc1, t, T)

                jerk_max = max([np.linalg.norm(jerk1), np.linalg.norm(jerk2)])
                acc_max = max([np.linalg.norm(acceleration1), np.linalg.norm(acceleration2), np.linalg.norm(acceleration3)])

                return acc_max, jerk_max
    
        def generateTrajectoryParametersRadius(self, blend_radius):
                
                self.duration_blend_list_cart.append(0)
                self.duration_blend_list_quat.append(0)
                for i in range(0, self.num_segments-1):
                        self.duration_blend_list_cart.append(self.min_blend_duration)
                        self.duration_blend_list_quat.append(self.min_blend_duration)
                self.duration_blend_list_cart.append(0)
                self.duration_blend_list_quat.append(0)
                
                for i in range(0, len(self.duration_blend_list_cart)):

                        if self.duration_blend_list_cart[i] > 0:
                                
                                first_segment = self.Segments[i-1]
                                second_segment = self.Segments[i]
                                first_segment_duration = first_segment.duration
                                second_segment_duration = second_segment.duration
                                
                                first_segment_dist = first_segment.dist
                                second_segment_dist = second_segment.dist
                                
                                velocity1 = (first_segment_dist/first_segment_duration)
                                velocity2 = (second_segment_dist/second_segment_duration)
                                
                                blend_time1 = first_segment_duration*0.5
                                if velocity1 > 0.0:
                                        blend_time1 = min(blend_radius/velocity1, first_segment_duration*0.5)
                                        
                                blend_time2 = second_segment_duration*0.5
                                if velocity2 > 0.0:
                                        blend_time2 = min(blend_radius/velocity2, second_segment_duration*0.5)
                                
                                blend_time = min(blend_time1, blend_time2)
                                
                                self.duration_blend_list_cart[i] = (2*blend_time)
                                #self.duration_blend_list_quat[i] = (2*blend_time)
                        
                                self.p0_list.append(first_segment.getPosition(first_segment_duration - blend_time))
                                self.p1_list.append(second_segment.getPosition(blend_time))

                                self.ang_v0_list.append(first_segment.getAngularVelocity())
                                self.ang_v1_list.append(second_segment.getAngularVelocity())

                                self.v0_list.append(first_segment.getVelocity(first_segment_duration - blend_time))
                                self.v1_list.append(second_segment.getVelocity(blend_time))

                                self.a0_list.append(first_segment.getAcceleration(first_segment_duration - blend_time))
                                self.a1_list.append(second_segment.getAcceleration(blend_time))
                        else:
                                self.p0_list.append(np.array([0,0,0]))
                                self.p1_list.append(np.array([0,0,0]))

                                self.ang_v0_list.append(np.array([0,0,0]))
                                self.ang_v1_list.append(np.array([0,0,0]))

                                self.v0_list.append(np.array([0,0,0]))
                                self.v1_list.append(np.array([0,0,0]))
                                self.a0_list.append(np.array([0,0,0]))
                                self.a1_list.append(np.array([0,0,0]))
                                
                cnt = 0               
                blend_phase_overlap = True
                blend_phase_overlap_quat = True
                while (blend_phase_overlap or max_acc_limit_violated_quat) or cnt < 10:
                        
                        cnt += 1
                        print("cnt: ", cnt)
                        blend_phase_overlap = False
                        for i in range(self.num_segments-1):

                                if 0.5*(self.duration_blend_list_quat[i] + self.duration_blend_list_quat[i+1]) > self.Segments[i].duration:

                                        self.Segments[i].duration *= 1.05  
                                
                                        print("blendphases overlapped")
                                        blend_phase_overlap = True

                                        seg = self.Segments[i]

                                        self.ang_v1_list[i] = (seg.getAngularVelocity())
                                        self.ang_v0_list[i+1] = (seg.getAngularVelocity())

                         
                        for i in range(0, len(self.duration_blend_list_cart)):

                                if self.duration_blend_list_cart[i] > 0:

                                        first_segment = self.Segments[i-1]
                                        second_segment = self.Segments[i]
                                        first_segment_duration = first_segment.duration
                                        second_segment_duration = second_segment.duration

                                        first_segment_dist = first_segment.dist
                                        second_segment_dist = second_segment.dist

                                        velocity1 = (first_segment_dist/first_segment_duration)
                                        velocity2 = (second_segment_dist/second_segment_duration)

                                        blend_time1 = first_segment_duration*0.5
                                        if velocity1 > 0.0:
                                                blend_time1 = min(blend_radius/velocity1, first_segment_duration*0.5)

                                        blend_time2 = second_segment_duration*0.5
                                        if velocity2 > 0.0:
                                                blend_time2 = min(blend_radius/velocity2, second_segment_duration*0.5)

                                        blend_time = min(blend_time1, blend_time2)

                                        self.duration_blend_list_cart[i] = (2*blend_time)
                                        

                                        self.p0_list[i] = (first_segment.getPosition(first_segment_duration - blend_time))
                                        self.p1_list[i] = (second_segment.getPosition(blend_time))

                                        self.ang_v0_list[i] = (first_segment.getAngularVelocity())
                                        self.ang_v1_list[i] = (second_segment.getAngularVelocity())

                                        self.v0_list[i] = (first_segment.getVelocity(first_segment_duration - blend_time))
                                        self.v1_list[i] = (second_segment.getVelocity(blend_time))

                                        self.a0_list[i] = (first_segment.getAcceleration(first_segment_duration - blend_time))
                                        self.a1_list[i] = (second_segment.getAcceleration(blend_time))
                                        
                                        
                        max_acc_limit_violated_quat = False 
                        for i in range(self.num_segments+1):
                                
                                if self.duration_blend_list_quat[i] > 0:

                                        max_angular_acc, max_angular_jerk = self.maxAccJerkCubic(self.ang_v0_list[i], self.ang_v1_list[i], np.array([0,0,0]), np.array([0,0,0]), self.duration_blend_list_quat[i]) 

                                        lamda_acc = max_angular_acc/self.ang_acc_max

                                        if max_angular_acc > self.ang_acc_max*1.01:
                                                print("acceleration limit for orientation interpolation violated!")
                                                max_acc_limit_violated_quat = True

                                        self.duration_blend_list_quat[i] *= lamda_acc

                                        if self.duration_blend_list_quat[i] < self.min_blend_duration:
                                                self.duration_blend_list_quat[i] = self.min_blend_duration


                                        self.ang_v0_list[i] = self.Segments[i-1].getAngularVelocity()
                                        self.ang_v1_list[i] = self.Segments[i].getAngularVelocity()

                                        max_angular_acc, max_angular_jerk = self.maxAccJerkCubic(self.ang_v0_list[i], self.ang_v1_list[i], np.array([0,0,0]), np.array([0,0,0]), self.duration_blend_list_quat[i]) 

                                        if max_angular_jerk > self.ang_jerk_max:
                                                self.duration_blend_list_quat[i] *= np.sqrt(max_angular_jerk/self.ang_jerk_max)

                                                if self.duration_blend_list_quat[i] < self.min_blend_duration:
                                                        self.duration_blend_list_quat[i] = self.min_blend_duration

                                                self.ang_v0_list[i] = self.Segments[i-1].getAngularVelocity()
                                                self.ang_v1_list[i] = self.Segments[i].getAngularVelocity()

                
                self.time_blend_start_cart.append(0)
                self.time_blend_start_quat.append(0)
                self.time_vector.append(0)

                for i in range(0,self.num_segments):
                        duration_sum = 0
                        for j in range(i+1):
                                duration_sum += self.Segments[j].duration
                        self.time_vector.append(duration_sum)
                        self.time_blend_start_cart.append(duration_sum - self.duration_blend_list_cart[i+1]*0.5)              
                        self.time_blend_start_quat.append(duration_sum - self.duration_blend_list_quat[i+1]*0.5)
                                
                                
                        
        def generateTrajectoryParameters(self):    
                
                self.duration_blend_list_cart.append(0)
                self.duration_blend_list_quat.append(0)
                for i in range(0, self.num_segments-1):
                        self.duration_blend_list_cart.append(self.min_blend_duration)
                        self.duration_blend_list_quat.append(self.min_blend_duration)
                self.duration_blend_list_cart.append(0)
                self.duration_blend_list_quat.append(0)
        
        
                for i in range(0, len(self.duration_blend_list_cart)):

                        if self.duration_blend_list_cart[i] > 0:
                                half_T_blend = self.duration_blend_list_cart[i]*0.5
                                first_segment = self.Segments[i-1]
                                second_segment = self.Segments[i]
                                first_segment_duration = first_segment.duration


                                self.p0_list.append(first_segment.getPosition(first_segment_duration - half_T_blend))
                                self.p1_list.append(second_segment.getPosition(half_T_blend))

                                self.ang_v0_list.append(first_segment.getAngularVelocity())
                                self.ang_v1_list.append(second_segment.getAngularVelocity())

                                self.v0_list.append(first_segment.getVelocity(first_segment_duration - half_T_blend))
                                self.v1_list.append(second_segment.getVelocity(half_T_blend))

                                self.a0_list.append(first_segment.getAcceleration(first_segment_duration - half_T_blend))
                                self.a1_list.append(second_segment.getAcceleration(half_T_blend))
                        else:
                                self.p0_list.append(np.array([0,0,0]))
                                self.p1_list.append(np.array([0,0,0]))

                                self.ang_v0_list.append(np.array([0,0,0]))
                                self.ang_v1_list.append(np.array([0,0,0]))

                                self.v0_list.append(np.array([0,0,0]))
                                self.v1_list.append(np.array([0,0,0]))
                                self.a0_list.append(np.array([0,0,0]))
                                self.a1_list.append(np.array([0,0,0]))

                
                cnt = 0               
                blend_phase_overlap = True
                blend_phase_overlap_quat = True
                max_acc_limit_violated_quat = True
                while (blend_phase_overlap or max_acc_limit_violated_cart or max_acc_limit_violated_quat) or cnt < 10:
                        
                        cnt += 1
                        print("cnt: ", cnt)
                        
                        blend_phase_overlap = False
                        for i in range(self.num_segments-1):

                                if 0.5*(self.duration_blend_list_cart[i] + self.duration_blend_list_cart[i+1]) > self.Segments[i].duration or 0.5*(self.duration_blend_list_quat[i] + self.duration_blend_list_quat[i+1]) > self.Segments[i].duration:

                                        self.Segments[i].duration *= 1.05  
                                        
                                        print("blendphases overlapped")
                                        blend_phase_overlap = True

                                        seg = self.Segments[i]
                                        self.p1_list[i] = (seg.getPosition(self.duration_blend_list_cart[i]*0.5))
                                        self.v1_list[i] = (seg.getVelocity(self.duration_blend_list_cart[i]*0.5))
                                        self.a1_list[i] = (seg.getAcceleration(self.duration_blend_list_cart[i]*0.5))

                                        self.p0_list[i+1] = (seg.getPosition(seg.duration - self.duration_blend_list_cart[i+1]*0.5))
                                        self.v0_list[i+1] = (seg.getVelocity(seg.duration - self.duration_blend_list_cart[i+1]*0.5))
                                        self.a0_list[i+1] = (seg.getAcceleration(seg.duration - self.duration_blend_list_cart[i+1]*0.5))

                                        self.ang_v1_list[i] = (seg.getAngularVelocity())
                                        self.ang_v0_list[i+1] = (seg.getAngularVelocity())



                        max_acc_limit_violated_cart = False  
                        max_acc_limit_violated_quat = False 
                        for i in range(self.num_segments+1):
                                
                                if self.duration_blend_list_cart[i] > 0:
                                        
                                        max_acc, max_jerk = self.maxAccJerkQuintic(self.p0_list[i], self.p1_list[i], self.v0_list[i], self.v1_list[i], self.a0_list[i], self.a1_list[i], self.duration_blend_list_cart[i]) 

                                        a_lim = max([self.a_max, np.linalg.norm(self.a0_list[i]), np.linalg.norm(self.a1_list[i])])
                                        lamda_acc = max_acc/a_lim

                                        if max_acc > a_lim*1.01:
                                                print("acceleration limit for translational interpolation violated!")
                                                max_acc_limit_violated_cart = True

                                        self.duration_blend_list_cart[i] *= lamda_acc

                                        if self.duration_blend_list_cart[i] < self.min_blend_duration:
                                                self.duration_blend_list_cart[i] = self.min_blend_duration

                                        seg1 = self.Segments[i-1]
                                        seg2 = self.Segments[i]

                                        half_T_blend = self.duration_blend_list_cart[i]*0.5

                                        self.p0_list[i] = (seg1.getPosition(seg1.duration - half_T_blend))
                                        self.p1_list[i] = (seg2.getPosition(half_T_blend))

                                        self.v0_list[i] = (seg1.getVelocity(seg1.duration - half_T_blend))
                                        self.v1_list[i] = (seg2.getVelocity(half_T_blend))

                                        self.a0_list[i] = (seg1.getAcceleration(seg1.duration - half_T_blend))
                                        self.a1_list[i] = (seg2.getAcceleration(half_T_blend))

                                        max_acc, max_jerk = self.maxAccJerkQuintic(self.p0_list[i], self.p1_list[i], self.v0_list[i], self.v1_list[i], self.a0_list[i], self.a1_list[i], self.duration_blend_list_cart[i]) 

                                        if max_jerk > self.j_max:
                                                self.duration_blend_list_cart[i] *= np.sqrt(max_jerk/self.j_max)

                                                if self.duration_blend_list_cart[i] < self.min_blend_duration:
                                                        self.duration_blend_list_cart[i] = self.min_blend_duration

                                                half_T_blend = self.duration_blend_list_cart[i]*0.5

                                                self.p0_list[i] = (seg1.getPosition(seg1.duration - half_T_blend))
                                                self.p1_list[i] = (seg2.getPosition(half_T_blend))

                                                self.v0_list[i] = (seg1.getVelocity(seg1.duration - half_T_blend))
                                                self.v1_list[i] = (seg2.getVelocity(half_T_blend))

                                                self.a0_list[i] = (seg1.getAcceleration(seg1.duration - half_T_blend))
                                                self.a1_list[i] = (seg2.getAcceleration(half_T_blend))


                                if self.duration_blend_list_quat[i] > 0:

                                        max_angular_acc, max_angular_jerk = self.maxAccJerkCubic(self.ang_v0_list[i], self.ang_v1_list[i], np.array([0,0,0]), np.array([0,0,0]), self.duration_blend_list_quat[i]) 

                                        lamda_acc = max_angular_acc/self.ang_acc_max

                               
                                        if max_angular_acc > self.ang_acc_max*1.01:
                                                print("acceleration limit for orientation interpolation violated!")
                                                max_acc_limit_violated_quat = True

                                        self.duration_blend_list_quat[i] *= lamda_acc

                                        if self.duration_blend_list_quat[i] < self.min_blend_duration:
                                                self.duration_blend_list_quat[i] = self.min_blend_duration


                                        self.ang_v0_list[i] = self.Segments[i-1].getAngularVelocity()
                                        self.ang_v1_list[i] = self.Segments[i].getAngularVelocity()

                                        max_angular_acc, max_angular_jerk = self.maxAccJerkCubic(self.ang_v0_list[i], self.ang_v1_list[i], np.array([0,0,0]), np.array([0,0,0]), self.duration_blend_list_quat[i]) 

                                        if max_angular_jerk > self.ang_jerk_max:
                                                self.duration_blend_list_quat[i] *= np.sqrt(max_angular_jerk/self.ang_jerk_max)

                                                if self.duration_blend_list_quat[i] < self.min_blend_duration:
                                                        self.duration_blend_list_quat[i] = self.min_blend_duration

                                                self.ang_v0_list[i] = self.Segments[i-1].getAngularVelocity()
                                                self.ang_v1_list[i] = self.Segments[i].getAngularVelocity()

                
                self.time_blend_start_cart.append(0)
                self.time_blend_start_quat.append(0)
                self.time_vector.append(0)

                for i in range(0,self.num_segments):
                        duration_sum = 0
                        for j in range(i+1):
                                duration_sum += self.Segments[j].duration
                        self.time_vector.append(duration_sum)
                        self.time_blend_start_cart.append(duration_sum - self.duration_blend_list_cart[i+1]*0.5)              
                        self.time_blend_start_quat.append(duration_sum - self.duration_blend_list_quat[i+1]*0.5)      


        
        
        def determineSegmentType(self, t, time_blend_start, T_blend_list):
                # first find in which area we are
                cnt = 0
                for i in range(0, len(time_blend_start)):
                        if t < time_blend_start[i]:
                                cnt = i-1
                                break;
                
                if t < 0:
                        cnt = 0
                
                if t >= time_blend_start[-1]:
                        cnt =  len(time_blend_start)-2

                if t > time_blend_start[cnt] and t <= time_blend_start[cnt] + T_blend_list[cnt]:
                        return "blend", cnt

                else:
                        return "lin", cnt
                
        
        
        def evaluate(self, t):
                seg_cart, cnt_cart = self.determineSegmentType(t, self.time_blend_start_cart, self.duration_blend_list_cart)
                seg_quat, cnt_quat = self.determineSegmentType(t, self.time_blend_start_quat, self.duration_blend_list_quat)
        
                if seg_cart == "blend":
                        dt = t - self.time_blend_start_cart[cnt_cart]
                        pos, vel, acc, jerk = self.quinticPolynomial(self.p0_list[cnt_cart], self.p1_list[cnt_cart], self.v0_list[cnt_cart], self.v1_list[cnt_cart], self.a0_list[cnt_cart], self.a1_list[cnt_cart], dt, self.duration_blend_list_cart[cnt_cart])

                if seg_cart == "lin":
                        dt = t - self.time_blend_start_cart[cnt_cart] - 0.5*self.duration_blend_list_cart[cnt_cart]
                        jerk = np.array([0,0,0])
                        acc = self.Segments[cnt_cart].getAcceleration(dt)
                        vel = self.Segments[cnt_cart].getVelocity(dt)
                        pos = self.Segments[cnt_cart].getPosition(dt)
            
                if seg_quat == "blend":
                        dt = t - self.time_blend_start_quat[cnt_quat]
                        log, angular_velocity, angular_acceleration, angular_jerk = self.cubicPolynomial(self.ang_v0_list[cnt_quat], self.ang_v1_list[cnt_quat], np.array([0,0,0]), np.array([0,0,0]), dt, self.duration_blend_list_quat[cnt_quat])

                        log -= self.ang_v0_list[cnt_quat]*0.5*self.duration_blend_list_quat[cnt_quat]

                        quaternion = Quaternion.exp(Quaternion(0,log[0], log[1], log[2])*0.5)*self.Segments[cnt_quat].q1

                if seg_quat == "lin":
                        dt = t - self.time_blend_start_quat[cnt_quat] - 0.5*self.duration_blend_list_quat[cnt_quat]
                        angular_jerk = np.array([0,0,0])
                        angular_acceleration = np.array([0,0,0])
                        angular_velocity = self.Segments[cnt_quat].getAngularVelocity()

                        log = angular_velocity*dt

                        quaternion = Quaternion.exp(Quaternion(0, log[0], log[1], log[2])*0.5)*self.Segments[cnt_quat].q1

        
                return pos, vel, acc, jerk, quaternion, angular_velocity, angular_acceleration, angular_jerk
        

        def evaluateDQ(self, t):
                seg_cart, cnt_cart = self.determineSegmentType(t, self.time_blend_start_cart, self.duration_blend_list_cart)
                seg_quat, cnt_quat = self.determineSegmentType(t, self.time_blend_start_quat, self.duration_blend_list_quat)
        
                if seg_cart == "blend":
                        dt = t - self.time_blend_start_cart[cnt_cart]
                        pos, vel, acc, jerk = self.quinticPolynomial(self.p0_list[cnt_cart], self.p1_list[cnt_cart], self.v0_list[cnt_cart], self.v1_list[cnt_cart], self.a0_list[cnt_cart], self.a1_list[cnt_cart], dt, self.duration_blend_list_cart[cnt_cart])

                if seg_cart == "lin":
                        dt = t - self.time_blend_start_cart[cnt_cart] - 0.5*self.duration_blend_list_cart[cnt_cart]
                        jerk = np.array([0,0,0])
                        acc = self.Segments[cnt_cart].getAcceleration(dt)
                        vel = self.Segments[cnt_cart].getVelocity(dt)
                        pos = self.Segments[cnt_cart].getPosition(dt)
            
                if seg_quat == "blend":
                        dt = t - self.time_blend_start_quat[cnt_quat]
                        log, angular_velocity, angular_acceleration, angular_jerk = self.cubicPolynomial(self.ang_v0_list[cnt_quat], self.ang_v1_list[cnt_quat], np.array([0,0,0]), np.array([0,0,0]), dt, self.duration_blend_list_quat[cnt_quat])

                        log -= self.ang_v0_list[cnt_quat]*0.5*self.duration_blend_list_quat[cnt_quat]

                        quaternion = Quaternion.exp(Quaternion(0,log[0], log[1], log[2])*0.5)*self.Segments[cnt_quat].q1

                if seg_quat == "lin":
                        dt = t - self.time_blend_start_quat[cnt_quat] - 0.5*self.duration_blend_list_quat[cnt_quat]
                        angular_jerk = np.array([0,0,0])
                        angular_acceleration = np.array([0,0,0])
                        angular_velocity = self.Segments[cnt_quat].getAngularVelocity()

                        log = angular_velocity*dt

                        quaternion = Quaternion.exp(Quaternion(0, log[0], log[1], log[2])*0.5)*self.Segments[cnt_quat].q1
                
                # construct pure quaternions from the computed accelerations and velocities
                w = Quaternion(0, *angular_velocity)
                w_dot = Quaternion(0, *angular_acceleration)
                
                t = Quaternion(0, *pos)
                v = Quaternion(0, *vel)
                a = Quaternion(0, *acc)
                
                # compute and return the respective dual quaternions
                dq = DualQuaternion.fromQuatPos(quaternion, pos)
               
                dq_dot = DualQuaternion(0.5*(w*dq.real), 0.5*(v + 0.5*t*w)*dq.real)
                
                dq_ddot = DualQuaternion(0.5*(w_dot + 0.5*w*w)*dq.real, 0.5*((a + 0.5*(v*w + t*w_dot)) + 0.5*(v + 0.5*t*w)*w)*dq.real)
                
               # dq_ddot = DualQuaternion(0.5*(w_dot*dq.real + w*dq_dot.real), 0.5*(w_dot*dq.dual + w*dq_dot.dual+a*dq.real + v*dq_dot.real))

                return dq, dq_dot, dq_ddot
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    