import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.DifferentialKinematics import DifferentialKinematics
from Simulation.QP_DifferentialKinematics import QP_DifferentialKinematics
from neura_dual_quaternions import DualQuaternion

class TaskExecutor:
        # Initialize the TaskExecutor class with a list of tasks, initial joint configuration (q), kinematics type, and method
        def __init__(self, task_list, init_q, fk_type, method = "classic", extended = False):
                # Store provided parameters and initialize time and status variables
                self.task_list = task_list  # List of tasks to be executed
                self.method = method  # Method for solving kinematics (classic or another specified method)
                self.extended = extended  # Flag for extended functionality (not used in this snippet)
                self.time = 0  # Current time
                self.time_orig = 0  # Original time (may differ from 'time' when scaling)
                self.done = False  # Flag indicating whether all tasks are done
                self.res = None  # Result variable for storing outcomes (not used in this snippet)
                self.time_scale = 1.0  # Time scaling factor
                
                 # Initialize kinematics objects based on the type
                self.fk = ForwardKinematics(fk_type)
                    
                 # Select and initialize appropriate differential kinematics solver based on the method
                if self.method == "classic":
                        self.idk = DifferentialKinematics(fk_type)
                else:
                        self.idk = QP_DifferentialKinematics(fk_type, self.method)
                
                # Initialize lists to store various metrics and results over time
                self.time_list = []
                self.error_norm_list = []
                self.gradient_list = []
                self.q_list = []
                self.q_dot_list = []
                self.q_dot_norm_list = []
                self.time_scale_list = []
                
                # Predetermined list of prediction times for future task evaluation
                self.pred_time_list = [0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1]
                
                
                # make sure we can predict the future state, set initial dual quaternion transformation of each task
                for i in range(0, len(self.task_list)):
                        task = self.task_list[i]
                        
                        # Initialize the dual quaternion transformation of each task based on its type and the previous task
                        # It essentially sets the starting state for each task
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
                               
                               
                # Create a time vector representing the cumulative time at the end of each task
                self.time_vector = [0]
                
                time_sum = 0
                for task in self.task_list:
                        time_sum += task.total_Time
                        self.time_vector.append(time_sum)
                

                # Initialize the joint variables and their derivatives
                self.q = init_q  # Initial joint positions
                self.q_dot = np.zeros(self.fk.dof)  # Initial joint velocities
                self.q_ddot = np.zeros(self.fk.dof)  # Initial joint accelerations
                
                # Initialize desired dual quaternion states for position and velocity
                self.x_des = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
                self.x_des_dot = DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
            

        # Method to determine the current task based on the time
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
        
        
        # Method to predict future Cartesian states of tasks based on the given prediction time
        def predictCartTasks(self, time_predict):
                
                cnt = self.getTaskCounter(time_predict)
                
                task = self.task_list[cnt]
                
                if task.type == "joint":
                        return task.x0, DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                if task.type == "wait":
                        return task.x0, DualQuaternion.basicConstructor(0,0,0,0, 0,0,0,0)
                else:
                        return task.evaluate(time_predict - self.time_vector[cnt])
                
        
        # Main method to update and execute tasks based on the time delta (dt)
        def run(self, dt):
                
                # update original and scaled time
                self.time_orig += dt
                self.time += self.time_scale*dt
                
                # Determine the current task index based on the scaled time
                cnt = self.getTaskCounter(self.time)
                task = self.task_list[cnt]
                
                error_norm = 0
                
                # If the current task is a 'joint' type, it involves joint motion
                if task.type == "joint":
                        # if initial joint position is not set, update q0 once
                        if not task.q0_set:
                                task.q0_set = True
                                task.q0 = self.q
                                
                        # Evaluate the task to get new joint positions and velocities
                        self.q, self.q_dot = task.evaluate(self.time - self.time_vector[cnt])
                        
                        # Update the desired spatial position using forward kinematics
                        self.x_des = self.fk.getFK(self.q)
                        error_norm = 0
                
                # For other types of tasks (not 'joint')
                else:
                        
                        # Evaluate the task to get desired spatial position and velocity
                        self.x_des, self.x_des_dot = task.evaluate(self.time - self.time_vector[cnt])
                        
                        # Get the real spatial position using forward kinematics based on joint positions
                        x_real = self.fk.getFK(self.q)
                        
                        # Calculate the error vector between desired and real positions
                        error = (self.x_des - x_real).asVector().flatten()
                        error_norm = np.linalg.norm(error)
                       
                        # Initialize the predicted direction vector for future states
                        pred_dir = np.ones(6)*0.5
                        # Iterate through predetermined prediction times
                        for i in range(len(self.pred_time_list)):
                                x, x_dot = self.predictCartTasks(self.time + self.pred_time_list[i])
                                dir_ = (2.0*x.inverse()*x_dot).as6Vector().flatten()
                                # Update the predicted direction
                                pred_dir += abs(dir_)
                                
                        # Normalize the predicted direction vector
                        pred_dir = pred_dir/np.linalg.norm(pred_dir)
                        
                        # Calculate new joint velocities and time scaling factor using appropriate differential kinematics method
                        if self.method == "classic":
                                self.q_dot, self.time_scale = self.idk.differential_kinematics_DQ(self.q, self.q_dot, self.x_des, self.x_des_dot)
                        else:
                                self.q_dot, self.time_scale = self.idk.quadratic_program(self.q, self.q_dot, self.x_des, self.x_des_dot, pred_dir)
                                
                        # Introduce a small amount of white noise to the joint velocities for robustness and simulation of real-world imperfections
                        white_noise_vector = np.random.normal(0, 1, self.fk.dof)
                        
                        # Update the joint positions with new velocities and white noise
                        self.q = self.q + (self.q_dot + 0.0001*white_noise_vector)*dt
               
                # Recording various metrics and status over time
                self.time_list.append(self.time_orig)
                self.q_list.append(self.q[:7])
                self.q_dot_list.append(self.q_dot[:7])
                self.q_dot_norm_list.append(np.linalg.norm(self.q_dot[:7]))
                self.gradient_list.append(self.idk.gradient)
                self.error_norm_list.append(error_norm)
                self.time_scale_list.append(self.time_scale) 
                
                # Check and update the 'done' status based on the time and task completion
                if self.time > self.time_vector[-1]:
                        self.done = True
        
