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
        def __init__(self, task_list, robot_type = "normal", method = "classic"):
                self.fk_type = robot_type
                self.method = method
                self.task_list = task_list
                self.fk = ForwardKinematics(robot_type)
                
        def start(self):
                running = True
                current_time = time.time()
                last_time = time.time()
                count = 0
                t = 0
                dt = 0
                s = 0

                env = Swift()
                env.launch(realtime=True)

                # Make a panda model and set its joint angles to the ready joint configuration
                maira = rtb.models.URDF.Maira7M()

                # Add the robot to the simulator
                env.add(maira)
                
                Tee = sg.Axes(length = 0.12, pose = self.fk.M.asTransformation())
                T_target = sg.Axes(length = 0.12, pose = self.fk.M.asTransformation())
                
                env.add(Tee)
                env.add(T_target)
                
                if self.fk_type == "extended":
                    init_q = np.array([0,0,0,0,0,0,0,0])
                else: 
                    init_q = np.array([0,0,0,0,0,0,0])
                task_executor = TaskExecutor(self.task_list, init_q, self.fk_type, self.method)
                
                dt = 0.005
                while running:

                        task_executor.run(dt)
                        
                        q = task_executor.q[:7]
                        maira.q = q
                        
                        q_ext = np.array([*q, 0])
                        Tee.T = self.fk.getFK(q_ext).asTransformation()
                        T_target.T = task_executor.x_des.asTransformation()

                        env.step(dt)
                        
                        if task_executor.done:
                                break;
                                running = False;
                                
                
                self.error_norm = task_executor.error_norm
                self.time_scale_list = task_executor.time_scale_list
                self.q_list = task_executor.q_list
                self.q_dot_list = task_executor.q_dot_list
                self.gradient_list = task_executor.gradient_list

    

