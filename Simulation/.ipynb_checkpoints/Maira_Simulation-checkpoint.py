import numpy as np
import time
from WaitTime import WaitTime
from MoveJoint import MoveJoint
from MoveLinear import MoveLinear
from CommandExecutor import CommandExecutor
from ForwardKinematics import ForwardKinematics

import matplotlib.pyplot as plt

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

print(maira)
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
#task_list = np.array([WaitTime(2)])

time_scaling = np.arange(0, 1.1, 0.1)

r_vec = np.array([0,1,0])
r_vec2 = np.array([0,0,1])
Q1 = Quaternion.fromAxisAngle(.9, r_vec)
Q2 = Quaternion.fromAxisAngle(4, r_vec2)
Q3 = Quaternion(1,0,0,0)
S1 = DualQuaternion.fromQuatPos(Q1, np.array([1,0,1]))
S2 = DualQuaternion.fromQuatPos(Q2, np.array([1,1,1]))
S3 = DualQuaternion.fromQuatPos(Q3, np.array([0,0,1.952]))

Tee1 = sg.Axes(length = 0.12, pose = S3.asTransformation())
Tee2 = sg.Axes(length = 0.12, pose = S3.asTransformation())

ax1 = sg.Axes(length = 0.08, base = S1.asTransformation())
ax2 = sg.Axes(length = 0.08, base = S2.asTransformation())
env.add(ax1)
env.add(ax2)
env.add(Tee1)
env.add(Tee2)

for s in time_scaling:
    S_int = S1*(S1.inverse()*S2).power(s)
    
    ax = sg.Axes(length = 0.02, base = S_int.asTransformation())
    env.add(ax)
    


command_executor = CommandExecutor(task_list, q1, 2, 7)
q_list = []
time_list = []
while running:
    
    if command_executor.done:
        break;
        running = False;
        
    command_executor.run(dt)
    q = command_executor.getPosition()
    maira.q = q
    
    q_list.append(q)
    time_list.append(t)
    
    T = forward_kinematcs.fk_screw_body(q)
    Tee1.T = T.asTransformation()
    #print(Tee.T)
    Tee2.T = maira.fkine(maira.q)
    #print(Tee.T)
    
    current_time = time.time()
    dt = (current_time - last_time)
    last_time = current_time
    count = count + 1
    t = t + dt
    
    env.step(dt)


plt.plot(time_list, q_list)
print("avg dt: ", 1000*(t/(1.0*count)), "ms")