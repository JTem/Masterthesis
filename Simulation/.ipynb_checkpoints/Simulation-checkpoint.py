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
        def __init__(self, task_list):
                self.task_list = task_list
                self.forward_kinematics = ForwardKinematics()
                
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
                
                Tee = sg.Axes(length = 0.12, pose = self.forward_kinematics.M.asTransformation())
                T_pred = sg.Axes(length = 0.18, pose = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0).asTransformation())
                T_pred2 = sg.Axes(length = 0.18, pose = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0).asTransformation())    
                
                env.add(Tee)
                #env.add(T_pred)
                #env.add(T_pred2)
                
                #command_executor = CommandExecutor(self.task_list, maira.q, 2, 7)
                task_executor = TaskExecutor(self.task_list, maira.q)
                
                dt = 0.01
                while running:


                        task_executor.run(dt)
                        q = task_executor.q
                        maira.q = q
                              
                        Tee.T = task_executor.x_des.asTransformation()
                        #T_pred.T = task_executor.x_predict.asTransformation()
                        #T_pred2.T = task_executor.x_predict2.asTransformation()         
                        
                        # current_time = time.time()
                        # dt = (current_time - last_time)
                        # last_time = current_time
                        # count = count + 1
                        # t = t + dt

                        env.step(dt)
                        
                        if task_executor.done:
                                break;
                                running = False;
                                
                
                self.error_norm = task_executor.error_norm
                self.q_dot_list = task_executor.q_dot_list

    

