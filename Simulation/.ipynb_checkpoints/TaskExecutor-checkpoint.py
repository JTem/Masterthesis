import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from neura_dual_quaternions import DualQuaternion

class TaskExecutor:
        
        def __init__(self, task_list, init_q):
                self.task_list = task_list
                
                self.time = 0
                self.done = False
                
                self.forward_kinematics = ForwardKinematics()
                self.differential_kinematics = DifferentialKinematics()
                
                # make sure we can predict the future state, set initial dual quaternion transformation of each task
                for i in range(0, len(self.task_list)):
                        task = self.task_list[i]
                               
                        if i == 0:
                                if task.type == "wait" or task.type == "joint":
                                        task.x0 = self.forward_kinematics.forward_kinematics(init_q)
                        
                        else:
                                last_task = self.task_list[i-1]
                                if task.type == "wait":
                                        
                                        if last_task.type == "joint":
                                                task.x0 = self.forward_kinematics.forward_kinematics(last_task.q1)
                                       
                                        elif last_task.type == "wait":
                                                task.x0 = last_task.x0
                                       
                                        else:
                                                task.x0, _ = last_task.evaluate(last_task.total_Time)
                               
                                if task.type == "joint":
                                        
                                        if last_task.type == "joint":
                                                task.x0 = self.forward_kinematics.forward_kinematics(last_task.q1)
                                       
                                        elif last_task.type == "wait":
                                                task.x0 = last_task.x0
                                       
                                        else:
                                                task.x0, _ = last_task.evaluate(last_task.total_Time)
                               
                               
                
                self.time_vector = [0]
                
                time_sum = 0
                for task in self.task_list:
                        time_sum += task.total_Time
                        self.time_vector.append(time_sum)
                
                print(self.time_vector)
                self.q = init_q
                self.q_dot = np.zeros(7)
                self.q_ddot = np.zeros(7)
                
                self.x_predict = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.x_predict2 = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                
                self.x_des = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.x_des_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                #self.x_ddot = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
            

        
        def getTaskCounter(self, time):
                cnt = 0
                for i in range(0, len(self.time_vector)):
                        if time < self.time_vector[i]:
                                cnt = i-1
                                break
                                
                if time < 0:
                        cnt = 0
                
                if time > self.time_vector[-1]:
                        cnt = len(self.time_vector)-2
                
                return cnt
        
        
        def predictCartTasks(self, cnt0, time_predict):
                
                cnt = self.getTaskCounter(time_predict)
                
                task = self.task_list[cnt]
                
                if task.type == "joint":
                        return task.x0, DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                if task.type == "wait":
                        return task.x0, DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                else:
                        return task.evaluate(time_predict - self.time_vector[cnt])
                
                
        def run(self, dt):
                
                self.time += dt
                
                cnt = self.getTaskCounter(self.time)
                task = self.task_list[cnt]
                
                       
                if task.type == "joint":
                        if not task.q0_set:
                                task.q0_set = True
                                task.q0 = self.q
                        self.q, self.q_dot = task.evaluate(self.time - self.time_vector[cnt])
                        self.x_des = self.forward_kinematics.forward_kinematics(self.q)
                        
                else:
                        self.x_des, self.x_des_dot = task.evaluate(self.time - self.time_vector[cnt])
                        self.x_predict, _ = self.predictCartTasks(cnt, self.time + 0.1)
                        self.x_predict2, _ = self.predictCartTasks(cnt, self.time + 0.2)
                        self.q_dot = self.differential_kinematics.quadratic_program_2(self.q, self.q_dot, self.x_des, self.x_des_dot)
                        self.q += self.q_dot*dt
                
                
                if self.time > self.time_vector[-1]:
                        self.done = True
        
