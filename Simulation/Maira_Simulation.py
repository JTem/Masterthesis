import numpy as np
import time
from WaitTime import WaitTime
from MoveJoint import MoveJoint
from MoveLinear import MoveLinear
from CommandExecutor import CommandExecutor
from ForwardKinematics import ForwardKinematics

import roboticstoolbox as rtb
import spatialmath as sm
import spatialgeometry as sg
import numpy as np
from swift import Swift


forward_kinematcs = ForwardKinematics()


def target_transformation(q):
        T = forward_kinematcs.forward_kinematics(q)
    
        return T

running = True
current_time = time.time()
last_time = time.time()
count = 0
t = 0
dt = 0
s = 0

env = Swift()
env.launch(realtime=False)

# Make a panda model and set its joint angles to the ready joint configuration
maira = rtb.models.URDF.Maira7M()

# Add the robot to the simulator
env.add(maira)


deg2rad = np.pi/180.0
q1 = np.zeros(7)

q2 = np.array([np.pi/180*0, -np.pi/180*60, -np.pi/180*0, -np.pi/180*60, -np.pi/180*0, -np.pi/180*60, -np.pi/180*0])
q3 = np.array([np.pi/180*0, np.pi/180*60, np.pi/180*0, np.pi/180*60, np.pi/180*0, np.pi/180*60, np.pi/180*0])

q4 = np.array([-np.pi/180*180, -np.pi/180*90, -np.pi/180*180, -np.pi/180*150, -np.pi/180*180, -np.pi/180*120, -np.pi/180*180])
q5 = np.array([np.pi/180*180, np.pi/180*90, np.pi/180*180, np.pi/180*150, np.pi/180*180, np.pi/180*120, np.pi/180*180])

qT1 = np.array([np.pi/180*-30, np.pi/180*0, np.pi/180*0, np.pi/180*90, np.pi/180*90, np.pi/180*30, 0])
qT2 = np.array([np.pi/180*30, np.pi/180*10, np.pi/180*0, np.pi/180*80, np.pi/180*0, np.pi/180*90, 0])

target_T1 = target_transformation(qT1)
target_T2 = target_transformation(qT2)

task_list = np.array([WaitTime(2), MoveJoint(q4, 3), MoveJoint(q5, 6), MoveJoint(q4, 6), MoveJoint(q2, 4), MoveJoint(q3, 4), MoveJoint(q2, 4), MoveJoint(q3, 4)])

command_executor = CommandExecutor(task_list, q1, 2, 7)
while running:
    
        if command_executor.done:
                break;
                running = False;

        command_executor.run(dt)
        q = command_executor.getPosition()
        maira.q = q

        current_time = time.time()
        dt = (current_time - last_time)
        last_time = current_time
        count = count + 1
        t = t + dt

        env.step(dt)
