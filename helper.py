import ipywidgets as widgets
import numpy as np
from collections import deque
import random
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from neura_dual_quaternions import Quaternion, DualQuaternion
np.set_printoptions(precision=2, suppress=True, linewidth=200, formatter={'float': '{:8.3f}'.format})

from Simulation.WaitTime import WaitTime
from Simulation.MoveJoint import MoveJoint
from Simulation.MoveLinear import MoveLinear
from Simulation.ForwardKinematics import ForwardKinematics

def plotIKResults(error_list_classic, error_list_DQ, time_classic, time_DQ, success_count_classic, success_count_DQ, num_eval):
        
        print("sucess rate classic IK: ", 100.0*success_count_classic/num_eval, "%")
        print("sucess rate DQ IK: ", 100.0*success_count_DQ/num_eval, "%")
        print("average time per iteration classic IK: ", time_classic/num_eval, "s")
        print("average time per iteration DQ IK: ", time_DQ/num_eval, "s")
        
        # Initialize a list to store the sum and count for each position
        sumsDQ = [0] * 22
        sum_itDQ = 0
        # Iterate through each list and position
        for lst in error_list_DQ:
                sum_itDQ += len(lst)
                for i in range(len(lst)):
                        sumsDQ[i] += lst[i]

        # Initialize a list to store the sum and count for each position
        sumsC = [0] * 22
        sum_itC = 0
        # Iterate through each list and position
        for lst in error_list_classic:
                sum_itC += len(lst)
                for i in range(len(lst)):
                        sumsC[i] += lst[i]
        
        # Calculate the average for each position
        average_error_DQ = np.array(sumsDQ) / num_eval
        average_error_C = np.array(sumsC) / num_eval

        print("average num of iterations for classic IK: ", sum_itC/num_eval)
        print("average num of iterations for DQ IK: ", sum_itDQ/num_eval)

        # Plotting error_norm
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        for i, sublist in enumerate(error_list_classic):
                plt.plot(sublist, color='g', alpha = 0.02)
        plt.plot(average_error_C, color = "m")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log')
        plt.title('Results Classic IK')
        plt.xlabel('Iterations')
        plt.ylabel('Error Norm')

        plt.subplot(2, 1, 2)
        for i, sublist in enumerate(error_list_DQ):
                plt.plot(sublist, color='b', alpha = 0.02)
        plt.plot(average_error_DQ, color = "r")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.yscale('log')
        plt.title('Results Dual Quaternion IK')
        plt.xlabel('Iterations')
        plt.ylabel('Error Norm')

        # Show the plots
        plt.tight_layout()
        plt.show()
        
        
def create_slider(name, start_val, min_val, max_val):
        slider_width = '95%'

        slider = widgets.FloatSlider(orientation='horizontal',description=name, value=start_val, min=min_val, max=max_val, step = 0.01, layout={'width': slider_width})
        return slider

def getRandomJointAngles():
        q1 = random.uniform(-np.pi, np.pi)
        q2 = random.uniform(-120*np.pi/180, 120*np.pi/180)
        q3 = random.uniform(-np.pi, np.pi)
        q4 = random.uniform(-150*np.pi/180, 150*np.pi/180)
        q5 = random.uniform(-np.pi, np.pi)
        q6 = random.uniform(-np.pi, np.pi)
        q7 = random.uniform(-np.pi, np.pi)
    
        q = np.array([q1, q2, q3, q4, q5, q6, q7])
        
        return q

def perturbJointAngles(q, perturbation_in_deg):
        
        qpert = []
        for i in range(7):
                qpert.append(random.uniform(-perturbation_in_deg*np.pi/180, perturbation_in_deg*np.pi/180))
        
        return q + np.array(qpert)


def deg2rad(deg):
        return np.pi/180.0*deg

def create_3d_plot(qr = Quaternion(1,0,0,0)):
    
        plt.ioff()
    
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        fig.canvas.header_visible = False
        fig.canvas.layout.min_height = '400px'
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        ax.set_facecolor('white')
    
        for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()

        start_point = [0, 0, 0]
        R_base = qr.asRotationMatrix()*1.5
        draw_frame(ax, start_point, R_base)

        return fig, ax

def create_gimble_lock_demo_plot():
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1])

        # Create the main 3D plot on the left
        fig = plt.figure(figsize=(8, 6))
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        fig.canvas.header_visible = False
        fig.canvas.layout.min_height = '400px'
        ax3d.set_xlim([-1, 1])
        ax3d.set_ylim([-1, 1])
        ax3d.set_zlim([-1, 1])
        ax3d.set_axis_off()
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_facecolor('white')

        # Create two 2D plots on the right
        ax2d_1 = fig.add_subplot(gs[0, 1])
        ax2d_2 = fig.add_subplot(gs[1, 1])
        ax2d_1.set_title("Quaternion log")
        ax2d_2.set_title("RPY angles")
        ax2d_1.set_facecolor('white')
        ax2d_2.set_facecolor('white')

        ax2d_2.set_ylim(-3.15, 3.15)
        ax2d_1.set_ylim(-3.15, 3.15)
        plt.tight_layout()

        # Parameters
        max_length = 20  # Maximum number of points to display
        x = np.linspace(0, max_length-1, max_length)

        deque1 = deque(maxlen=max_length)
        deque1.extend([0]*max_length)
        
        deque2 = deque(maxlen=max_length)
        deque2.extend([0]*max_length)

        deque3 = deque(maxlen=max_length)
        deque3.extend([0]*max_length)

        line1, = ax2d_2.plot(x, deque1, linewidth = 2)
        line2, = ax2d_2.plot(x, deque2, linewidth = 2)
        line3, = ax2d_2.plot(x, deque3, linewidth = 2)

        deque4 = deque(maxlen=max_length)
        deque4.extend([0]*max_length)

        deque5 = deque(maxlen=max_length)
        deque5.extend([0]*max_length)

        deque6 = deque(maxlen=max_length)
        deque6.extend([0]*max_length)

        line4, = ax2d_1.plot(x, deque4, linewidth = 2)
        line5, = ax2d_1.plot(x, deque5, linewidth = 2)
        line6, = ax2d_1.plot(x, deque6, linewidth = 2)
        
        lines = [line1, line2, line3, line4, line5, line6]
        deques = [deque1, deque2, deque3, deque4, deque5, deque6]
        
        return fig, ax3d, lines, deques

def update_gimble_lock_demo(lines, deques, roll, pitch, yaw, log):
        deques[0].append(roll)
        deques[1].append(pitch)
        deques[2].append(yaw)
        deques[3].append(log.x)
        deques[4].append(log.y)
        deques[5].append(log.z)

        lines[0].set_ydata(deques[0])
        lines[1].set_ydata(deques[1])
        lines[2].set_ydata(deques[2])

        lines[3].set_ydata(deques[3])
        lines[4].set_ydata(deques[4])
        lines[5].set_ydata(deques[5])
        
def draw_frame(ax, start_point, R):
    
        x_axis = ax.quiver(*start_point, *R[:,0], arrow_length_ratio = 0.1, linewidth = 1, color='r')
        y_axis = ax.quiver(*start_point, *R[:,1], arrow_length_ratio = 0.1, linewidth = 1, color='g')
        z_axis = ax.quiver(*start_point, *R[:,2], arrow_length_ratio = 0.1, linewidth = 1, color='b')
        return x_axis, y_axis, z_axis

def create_slider(name, start_val, min_val, max_val):
        slider_width = '98%'

        slider = widgets.FloatSlider(orientation='horizontal',description=name, value=start_val, min=min_val, max=max_val, step = 0.01, layout={'width': slider_width})
        return slider

def spherical_coordinates(azimuth, elevation):
        x = np.sin(np.pi/2 -elevation) * np.cos(azimuth)
        y = np.sin(np.pi/2 -elevation) * np.sin(azimuth)
        z = np.cos(np.pi/2 -elevation)
        vector = np.array([x, y, z])
        return vector

def create_textbox(name):
        slider_width = '98%'
        display = widgets.Text(description=name, value='', layout={'width': slider_width})
        return display

def create_quiver(ax, start_point, direction, w, c, l):
        quiver = ax.quiver(*start_point, *direction, arrow_length_ratio=0.1, linewidth = w, color = c, label = l)
        return quiver
    
def rpy_from_R(R):  
        # calculation of roll pitch yaw angles from rotation matrix
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        roll = np.arctan2(R[2, 1], R[2, 2])

        return roll, pitch, yaw

def Rot_rpy(roll, pitch, yaw):
    
        # Roll matrix
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

        # Pitch matrix
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

        # Yaw matrix
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

        # Combined rotation matrix
        R = Rz@Ry@Rx
        return R