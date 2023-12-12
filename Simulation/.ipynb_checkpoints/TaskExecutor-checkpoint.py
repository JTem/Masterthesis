import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from Simulation.MPC_DifferentialKinematics import MPC_DifferentialKinematics
from Simulation.QP_DifferentialKinematics import QP_DifferentialKinematics
from Simulation.QP_DifferentialKinematicsExtended import QP_DifferentialKinematicsExtended
from neura_dual_quaternions import DualQuaternion

class TaskExecutor:
        
        def __init__(self, task_list, init_q, fk_type):
                self.task_list = task_list
                
                self.time = 0
                self.done = False
                
                self.res = None
                self.time_scale = 1.0
                
                
                self.fk = ForwardKinematics(fk_type)
                self.differential_kinematics = DifferentialKinematics()
                if fk_type == "extended":
                    print("extended!!!!!!!!!")
                    self.qp_differential_kinematics = QP_DifferentialKinematicsExtended(fk_type)
                else:
                        self.qp_differential_kinematics = QP_DifferentialKinematics(fk_type)
                
                self.error_norm = []
                self.gradient_list = []
                self.q_list = []
                self.q_dot_list = []
                self.time_scale_list = []
                #self.pred_time_list = [0.9, 1.1, 1.3, 1.5, 1.7, 2.0, 2.2, 2.5, 2.7, 2.9, 3.0, 3.3, 3.5]
                self.pred_time_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1]#, 3.3, 3.5, 3.7, 4.0, 4.2, 4.4, 4.6, 5.0]
                self.use_mpc = False
                        
                self.N = 10
                self.Nu = 10
                self.dof = 7
                Ts0 = 0.002
                Ts_lin_fact = 2
                Ts_quat_fact = 2
                weight_x = 2
                weight_u = 2
                weight_s = 10_000_000
                weight_du = 0.0000001
                
                joint_limit = 3.14*np.ones(self.dof)
                velocity_limit = 3.14*np.ones(self.dof)
                acceleration_limit = 100*np.ones(self.dof)
                
                self.mpc = MPC_DifferentialKinematics(self.N, self.Nu, self.dof, Ts0, Ts_lin_fact, Ts_quat_fact, weight_x, weight_u, weight_s, weight_du, joint_limit, velocity_limit, acceleration_limit)
                
                
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
                
                self.time += self.time_scale*dt
                #print(self.time_scale)
                
                cnt = self.getTaskCounter(self.time)
                task = self.task_list[cnt]
                
                if task.type == "joint":
                        self.mpc.is_seeded = False
                        if not task.q0_set:
                                task.q0_set = True
                                task.q0 = self.q
                        self.q, self.q_dot = task.evaluate(self.time - self.time_vector[cnt])
                        self.x_des = self.fk.getFK(self.q)
                        
                else:
                        
                        if self.use_mpc and not self.mpc.is_seeded:
                                
                                x = self.fk.getFK(self.q)
                                J = self.fk.getSpaceJacobian(self.q)
                                J_H = 0.5*x.as_mat_right()@J
                                
                                ref_list = []
                                for i in range(self.mpc.Nu):
                                        _, x_dot = self.predictCartTasks(self.time + self.mpc.dt_vector[i])
                                        ref_list.append(x_dot.asVector().flatten())
                                
                                self.res = self.mpc.seed(self.q, self.q_dot, J_H, ref_list)
                        
                        if self.use_mpc:
                                
                                if not self.mpc.is_seeded:
                                        x = self.fk.getFK(self.q)
                                        J = self.fk.getSpaceJacobian(self.q)
                                        J_H = 0.5*x.as_mat_right()@J

                                        ref_list = []
                                        for i in range(self.mpc.Nu):
                                                _, x_dot = self.predictCartTasks(self.time + self.mpc.dt_vector[i])
                                                ref_list.append(x_dot.asVector().flatten())

                                        #print(ref_list)
                                        self.res = self.mpc.seed(self.q, self.q_dot, J_H, ref_list)
                                        #print(self.res)
                                
                                x = self.fk.getFK(self.q)
                                J = self.fk.getSpaceJacobian(self.q)
                                J_H = 0.5*x.as_mat_right()@J
                                
                                J_list = []
                                for i in range(self.Nu):
                                        # q_pred = self.res[self.dof*i:self.dof*i + self.dof]
                                        # x = self.forward_kinematics.forward_kinematics(q_pred)
                                        # J = self.forward_kinematics.jacobian(q_pred)
                                        # J_H = 0.5*x.as_mat_right()@J
                                        J_list.append(J_H)
                                        
                                ref_list = []
                                for i in range(self.mpc.Nu):
                                        _, x_dot = self.predictCartTasks(self.time + self.mpc.dt_vector[i])
                                        ref_list.append(x_dot.asVector().flatten())
                                
                                self.x_des, _ = task.evaluate(self.time - self.time_vector[cnt])
                                x_real = self.fk.getFK(self.q)
                                #print(ref_list)
                                #print(ref_list[0] + (self.x_des - x_real).asVector().flatten())
                                
                                error = (self.x_des - x_real).asVector().flatten()
                                ref_list[0] = ref_list[0]  + 2.0 * error
                                
                                self.error_norm.append(np.linalg.norm(error))
                                
                     
                                self.res = self.mpc.update(self.q, self.q_dot, J_list, ref_list)
                                self.q_dot = self.res[self.N*self.dof:self.N*self.dof + self.dof]
                                
                                self.q_dot_list.append(self.q_dot)
                          
                        
                        else:
                                self.x_des, self.x_des_dot = task.evaluate(self.time - self.time_vector[cnt])
                                
                                self.x_des, _ = task.evaluate(self.time - self.time_vector[cnt])
                                x_real = self.fk.getFK(self.q)
                                
                                error = (self.x_des - x_real).asVector().flatten()
                                self.error_norm.append(np.linalg.norm(error))
                                
                                pred_dir = np.ones(6)*0.5
                                #pred_dir = np.array([1,1,1,0.1,0.1,0.1])*0.02
                                for i in range(len(self.pred_time_list)):
                                        x, x_dot = self.predictCartTasks(self.time + self.pred_time_list[i])
                                        dir_ = (2.0*x.inverse()*x_dot).as6Vector().flatten()
                                        #dir_ = (2.0*x_dot*x.inverse()).as6Vector().flatten()
                                        # if np.linalg.norm(dir_) > 1e-6:
                                        #         dir_ = np.abs(dir_)/np.linalg.norm(dir_)
                                        
                                        pred_dir += abs(dir_)
                                        
                                if np.linalg.norm(pred_dir) > 1e-8:
                                        pred_dir = np.abs(pred_dir)/np.linalg.norm(pred_dir)
                                
                                #pred_dir = np.array([0,0,1,0.1,1,0])
                                #pred_dir = np.abs(pred_dir)/np.linalg.norm(pred_dir)
                                #self.q_dot = self.differential_kinematics.quadratic_program_1(self.q, self.q_dot, self.x_des, self.x_des_dot)
                                self.q_dot, self.time_scale = self.qp_differential_kinematics.quadratic_program(self.q, self.q_dot, self.x_des, self.x_des_dot, pred_dir)
                                
                                
                                    
                        self.q = self.q + self.q_dot*dt
                        
                        self.q_list.append(self.q[:7])
                        self.q_dot_list.append(self.q_dot[:7])
                        self.gradient_list.append(self.qp_differential_kinematics.gradient)
                        self.time_scale_list.append(self.time_scale) 
                
                if self.time > self.time_vector[-1]:
                        self.done = True
        
