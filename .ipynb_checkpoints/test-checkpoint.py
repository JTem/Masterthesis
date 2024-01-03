import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics
from Simulation.InverseKinematics import InverseKinematics
import matplotlib.pyplot as plt
import time

min_lim = np.array([-np.pi, -120*np.pi/180, -150*np.pi/180, -np.pi, -np.pi, -np.pi])
max_lim = np.array([np.pi, 120*np.pi/180, 150*np.pi/180, np.pi, np.pi, np.pi])

fk = ForwardKinematics() 
ik = InverseKinematics(min_lim, max_lim)

num_eval = 1000
success_count_classic = 0
success_count_DQ = 0
error_list_DQ = []
error_list_classic = []

time_classic = 0
time_DQ = 0
for i in range(num_eval):
    q_guess = getRandomJointAngles()
    
    x_target = fk.getFK(perturbJointAngles(q_guess, 85))
    
    start_time_classic = time.perf_counter()
    q_sol_classic, error_norm_classic, condition_classic, success_classic = ik.getIK_classic(x_target, q_guess, 0.001)
    end_time_classic = time.perf_counter()
    
    start_time_DQ = time.perf_counter()
    q_sol_DQ, error_norm_DQ, condition_DQ, success_DQ = ik.getIK_DQ(x_target, q_guess, 0.001)
    end_time_DQ = time.perf_counter()
    
    time_classic = time_classic + (end_time_classic - start_time_classic)
    time_DQ = time_DQ + (end_time_DQ - start_time_DQ)
    
    error_list_DQ.append(error_norm_DQ)
    error_list_classic.append(error_norm_classic)
    
    if success_classic:
        success_count_classic += 1
    if success_DQ:
        success_count_DQ += 1
        

plotIKResults(error_list_classic, error_list_DQ, time_classic, time_DQ, success_count_classic, success_count_DQ, num_eval)