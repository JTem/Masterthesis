import numpy as np
import time
from Simulation.WaitTime import WaitTime
from Simulation.MoveJoint import MoveJoint
from Simulation.MoveLinear import MoveLinear
from Simulation.TaskExecutor import TaskExecutor
from Simulation.ForwardKinematics import ForwardKinematics

import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import numpy as np
from swift import Swift
from neura_dual_quaternions import Quaternion, DualQuaternion

class Simulation:
        # Initialize the Simulation class with a list of tasks, robot type, and method for kinematics
        def __init__(self, task_list, robot_type = "normal", method = "classic"):
                self.fk_type = robot_type # Type of robot for kinematics (normal/extended)
                self.method = method # Method for solving kinematics (classic/qp/qp_yoshikawa))
                self.task_list = task_list
                self.fk = ForwardKinematics(robot_type) # Forward kinematics object for the robot type
        
        # Start the simulation
        def start(self):
                running = True
                current_time = time.time()
                last_time = time.time()
                
                # Initializing time, step time, and a counter for the simulation
                count = 0
                t = 0
                dt = 0
                s = 0
                
                # Create and launch a Swift simulator environment
                env = Swift()
                env.launch(realtime=True)

                # Create a robot model (here it's named Maira7M) and add it to the simulator
                maira = rtb.models.URDF.Maira7M()

                # Add the robot to the simulator
                env.add(maira)
                
                # Initialize the transformation objects for robot's end-effector (Tee) and the target (T_target)
                Tee = sg.Axes(length = 0.12, pose = self.fk.M.asTransformation())
                T_target = sg.Axes(length = 0.12, pose = self.fk.M.asTransformation())
                env.add(Tee)
                env.add(T_target)
                
                # Initialize the robot's joint configurations based on the robot type
                if self.fk_type == "extended":
                        init_q = np.array([0,0,0,0,0,0,0,0])
                else: 
                        init_q = np.array([0,0,0,0,0,0,0])
                
                # Create a TaskExecutor Object
                task_executor = TaskExecutor(self.task_list, init_q, self.fk_type, self.method)
                
                # Set the time step for the simulation (here 10ms because python is slow)
                dt = 0.01
                while running:
                        
                        # Execute tasks in the task executor
                        task_executor.run(dt)
                        
                        # Update the robot model's joint configuration
                        q = task_executor.q[:7]
                        maira.q = q
                        
                        # Update the transformation of the robot's end-effector and target based on the task execution
                        q_ext = np.array([*q, 0])
                        Tee.T = self.fk.getFK(q_ext).asTransformation()
                        T_target.T = task_executor.x_des.asTransformation()
                        
                        # Step the simulation environment forward
                        env.step(dt)
                        
                        # Check if the task execution is complete
                        if task_executor.done:
                                break;
                                running = False;
                                
                # Store various metrics and results from the task executor for analysis
                self.error_norm_list = task_executor.error_norm_list
                self.time_scale_list = task_executor.time_scale_list
                self.time_list = task_executor.time_list
                self.q_list = task_executor.q_list
                self.q_dot_list = task_executor.q_dot_list
                self.q_dot_norm_list = task_executor.q_dot_norm_list
                self.gradient_list = task_executor.gradient_list

    

