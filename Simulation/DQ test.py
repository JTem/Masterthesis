from neura_dual_quaternions import Quaternion, DualQuaternion
import numpy as np
import matplotlib.pyplot as plt

vec = np.array([0,0,1])
angle = 0

vec2 = angle*vec
Q = Quaternion.fromAxisAngle(angle, vec)

pos = np.array([0,0,0])
DQ = DualQuaternion.fromQuatPos(Q, pos)

print(DQ)

log = DQ.log()
print(log)

exp = DualQuaternion.exp(log)
print(exp)




# theta_vec = np.arange(0,0.2, 0.000001)
# print(theta_vec)
# y = []
# for theta in theta_vec:
#     if theta < 0.001:
#         frac_sin_theta_theta = 0.5 - theta**2/48.0 + theta**4/3840
#     else:
#         frac_sin_theta_theta = np.sin(0.5*theta)/theta
    
#     test = (np.cos(theta) - np.sin(theta)/theta)/(theta)**2
#     y.append(test)

    

# plt.plot(theta_vec, y)










































