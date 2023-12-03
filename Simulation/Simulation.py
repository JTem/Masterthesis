import numpy as np
import time
from Simulation.WaitTime import WaitTime
from Simulation.MoveJoint import MoveJoint
from Simulation.MoveLinear import MoveLinear
from Simulation.CommandExecutor import CommandExecutor
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
                T_target = sg.Axes(length = 0.18, pose = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0).asTransformation())
                                   
                env.add(Tee)
                env.add(T_target)
                                   
                command_executor = CommandExecutor(self.task_list, maira.q, 2, 7)
                dt = 0.005
                while running:

                        if command_executor.done:
                                break;
                                running = False;

                        command_executor.run(dt)
                        q = command_executor.getPosition()
                        maira.q = q
                              
                        Tee.T = self.forward_kinematics.forward_kinematics(q).asTransformation()
                        T_target.T = command_executor.getCartesianTarget().asTransformation()
                                   
                        # current_time = time.time()
                        # dt2 = (current_time - last_time)
                        # print(dt2)
                        # last_time = current_time
                        # count = count + 1
                        # t = t + dt

                        env.step(dt)

    

