import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from Simulation.QP_DifferentialKinematics import QP_DifferentialKinematics
from neura_dual_quaternions import DualQuaternion

class TaskExecutor:
        
        def __init__(self, task_list, init_q, fk_type, method = "classic", extended = False):
                self.task_list = task_list
                self.method = method
                self.extended = extended
                self.time = 0
                self.time_orig = 0
                self.done = False
                
                self.res = None
                self.time_scale = 1.0
                
                
                self.fk = ForwardKinematics(fk_type)
            
                if self.method == "classic":
                        self.idk = DifferentialKinematics(fk_type)
                else:
                        self.idk = QP_DifferentialKinematics(fk_type, self.method)
               
                
                self.time_list = []
                self.error_norm_list = []
                self.gradient_list = []
                self.q_list = []
                self.q_dot_list = []
                self.q_dot_norm_list = []
                self.time_scale_list = []
                self.pred_time_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1]
                
                
                # make sure we can predict the future state, set initial dual quaternion transformation of each task
                for i in range(0, len(self.task_list)):
                        task = self.task_list[i]
                               
                        if i == 0:
                                if task.type == "wait" or task.type == "joint":
                                        task.x0 = self.fk.getFK(init_q)
                        
                        else:
                                last_task = self.task_list[i-1]
                                if task.type == "wait":
                                        
                                        if last_task.type == "joint":
                                                task.x0 = self.fk.getFK(last_task.q1)
                                       
                                        elif last_task.type == "wait":
                                                task.x0 = last_task.x0
                                       
                                        else:
                                                task.x0, _ = last_task.evaluate(last_task.total_Time)
                               
                                if task.type == "joint":
                                        
                                        if last_task.type == "joint":
                                                task.x0 = self.fk.getFK(last_task.q1)
                                       
                                        elif last_task.type == "wait":
                                                task.x0 = last_task.x0
                                       
                                        else:
                                                task.x0, _ = last_task.evaluate(last_task.total_Time)
                               
                               
                
                self.time_vector = [0]
                
                time_sum = 0
                for task in self.task_list:
                        time_sum += task.total_Time
                        self.time_vector.append(time_sum)
                

                self.q = init_q
                self.q_dot = np.zeros(self.fk.dof)
                self.q_ddot = np.zeros(self.fk.dof)
                
                self.x_des = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.x_des_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
            

        
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
        
        
        def predictCartTasks(self, time_predict):
                
                cnt = self.getTaskCounter(time_predict)
                
                task = self.task_list[cnt]
                
                if task.type == "joint":
                        return task.x0, DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                if task.type == "wait":
                        return task.x0, DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                else:
                        return task.evaluate(time_predict - self.time_vector[cnt])
                
                
        def run(self, dt):
                
                self.time_orig += dt
                self.time += self.time_scale*dt
               
                cnt = self.getTaskCounter(self.time)
                task = self.task_list[cnt]
                error_norm = 0
                if task.type == "joint":
                        if not task.q0_set:
                                task.q0_set = True
                                task.q0 = self.q
                        self.q, self.q_dot = task.evaluate(self.time - self.time_vector[cnt])
                        self.x_des = self.fk.getFK(self.q)
                        error_norm = 0
                        
                else:
                        self.x_des, self.x_des_dot = task.evaluate(self.time - self.time_vector[cnt])

                        x_real = self.fk.getFK(self.q)

                        error = (self.x_des - x_real).asVector().flatten()
                        error_norm = np.linalg.norm(error)
                       

                        pred_dir = np.ones(6)*0.5
                        for i in range(len(self.pred_time_list)):
                                x, x_dot = self.predictCartTasks(self.time + self.pred_time_list[i])
                                dir_ = (2.0*x.inverse()*x_dot).as6Vector().flatten()
                                pred_dir += abs(dir_)

                        pred_dir = pred_dir/np.linalg.norm(pred_dir)
                        
                        if self.method == "classic":
                                self.q_dot, self.time_scale = self.idk.differential_kinematics_DQ(self.q, self.q_dot, self.x_des, self.x_des_dot)
                        else:
                                self.q_dot, self.time_scale = self.idk.quadratic_program(self.q, self.q_dot, self.x_des, self.x_des_dot, pred_dir)
                                
                                
                        white_noise_vector = np.random.normal(0, 1, self.fk.dof)
                        self.q = self.q + (self.q_dot + 0.0001*white_noise_vector)*dt
                
                self.time_list.append(self.time_orig)
                self.q_list.append(self.q[:7])
                self.q_dot_list.append(self.q_dot[:7])
                self.q_dot_norm_list.append(np.linalg.norm(self.q_dot[:7]))
                self.gradient_list.append(self.idk.gradient)
                self.error_norm_list.append(error_norm)
                self.time_scale_list.append(self.time_scale) 
                
                if self.time > self.time_vector[-1]:
                        self.done = True
        
