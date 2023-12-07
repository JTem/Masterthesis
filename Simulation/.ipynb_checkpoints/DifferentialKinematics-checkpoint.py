import numpy as np
from neura_dual_quaternions import Quaternion, DualQuaternion
from Simulation.ForwardKinematics import ForwardKinematics
import osqp
import scipy.sparse as sp

class DifferentialKinematics:
        def __init__(self):
                self.forward_kinematics = ForwardKinematics()
                self.q_dot_last = np.zeros(7)
                self.J_last = np.zeros((6,7))
                self.osqp_settings = {
                    'alpha': 1.0,
                    'verbose': False,
                    'max_iter': 1000,
                    'eps_abs': 1e-3
                }
        
        def manipulability(self, q):
                J = self.forward_kinematics.jacobian_body(q)
                
                return np.sqrt(max(0, np.linalg.det(J@J.T)))
        
        def manipulability_gradient(self, q):
                
                w0 = self.manipulability(q)
                h = 0.00001
                grad = []
                for i in range(7):
                        dq = np.zeros(7)
                        dq[i] = h
                        grad.append((self.manipulability(q+dq) -w0)/h)
                        
                return np.array(grad)
        
        def dir_manipulability_gradient2(self, q):
                
                J = self.forward_kinematics.jacobian_body(q)
                H = self.forward_kinematics.hessian(q)
                W = np.diag(np.array([0,0,0,0,0,0]))
                
                #manipulability =  np.sqrt(max(0, np.linalg.det(J@J.T)))
                b = np.linalg.inv(J@J.T + np.eye(6)*0.05)
                Jm = np.zeros(7)
                for i in range(7):
                        c = W@J@H[:,:,i].T
                        Jm[i] = (c.flatten("F").T)@b.flatten("F")
                        
                return Jm
        
        def dir_manipulability(self, q, v):
                Jb = self.forward_kinematics.jacobian_body(q)
                U, S, V = np.linalg.svd(Jb@Jb.T)
                
                sum_w = 0
                for i in range(6):
                        sum_w += abs(S[i]*(np.dot(v, U[:,i])))
                        
                return sum_w
        
        def dir_manipulability_gradient(self, q, v):
                
                w0 = self.dir_manipulability(q, v)
                h = 0.0001
                grad = []
                for i in range(7):
                        dq = np.zeros(7)
                        dq[i] = h
                        grad.append((self.dir_manipulability(q+dq, v) - w0)/h)
                
                return np.array(grad)
        
        
        def differential_kinematics(self, q, q_dot, DQd, DQd_dot):
                x = self.forward_kinematics.forward_kinematics(q)
                
                x_error = DQd*x.inverse()
                pos_error = x_error.getPosition()
                o_error = x_error.real.log().getVector()

                Omega = DQd_dot*DQd.inverse()*2.0
                x_dot = Omega.as6Vector()

                error = np.array([*o_error.flatten(), *pos_error.flatten()])
                J = self.forward_kinematics.jacobian6(q)
                
                kp = 20
                vel = x_dot.flatten() + kp*error
                pinv = np.linalg.pinv(J)
                self.gradient = self.dir_manipulability_gradient2(q)
                q_dot_ = pinv@vel.flatten() + 5.0*(np.eye(7)-pinv@J)@self.gradient
                
                #print(self.manipulability_gradient(q))
                return q_dot_.flatten()
        
    
        def differential_kinematics_DQ(self, q, q_dot, DQd, DQd_dot):
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                
                lamda = 0.000001
                I = np.eye(J_H.shape[1])
                J_pinv = np.linalg.inv(J_H.T @ J_H + lamda*I) @ J_H.T
                

                kp = 20

                vel = DQd_dot.asVector() + kp*error
                q_dot_ = J_pinv@vel.flatten()

                return q_dot_.flatten()
        
        def qrmc_DQ(self, q, q_dot, DQd, DQd_dot, dt):
                x = self.forward_kinematics.forward_kinematics(q)
                        
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                if np.allclose(np.zeros(7), q_dot):
                        qd = np.ones*0.01
                else:
                        qd = q_dot
                        
                J_dot = self.forward_kinematics.jacobian_dot(q, qd)
                
                J_H = 0.5*DQd.as_mat_right()@J
                J_dot_H = 0.5*DQd.as_mat_right()@J_dot
                
                Jhat = 0.5*DQd.as_mat_right()@(J + J_dot)
         
                
                kp = 0
                vel = DQd_dot.asVector() + kp*error

                delta_q_dot = J_pinv@(J@q_dot + 0.5*J_dot@q_dot - vel.flatten())
 
                q_dot -= delta_q_dot*dt
                
                return qd - delta_q_dot
        
        def qrmc(self, q, q_dot, DQd, DQd_dot, dt):
                x = self.forward_kinematics.forward_kinematics(q)
                
                x_error = DQd*x.inverse()

                pos_error = x_error.getPosition()
                
                o_error = x_error.real.log().getVector()

                Omega = DQd_dot*DQd.inverse()*2.0
                x_dot = Omega.as6Vector()
                
                #print(x_dot.flatten())
                error = np.array([*o_error.flatten(), *pos_error.flatten()])
                
                J = self.forward_kinematics.jacobian6(q)
                J2 = self.forward_kinematics.jacobian6(q + q_dot*dt)
                
                Jd = (J2 - J)/dt
                
                if np.allclose(np.zeros(7), q_dot):
                        qd = np.linalg.pinv(J) @ x_dot.flatten()
                else:
                        qd = q_dot
        
                #H = self.forward_kinematics.hessian(q)
                #Hq = np.tensordot(H, qd, axes=([2], [0]))
                
                J_dot = self.forward_kinematics.jacobian_dot6(q, qd)
                #print(J_dot)
                
                Jhat = (J + J_dot)
                
                kp = 2
                g = J@qd + 0.5*J_dot@qd - (x_dot.flatten() + kp*error)
                
                delta_qd = np.linalg.pinv(Jhat)@g
                
                
                return qd - delta_qd
        
        def quadratic_program_1(self, q, q_dot, DQd, DQd_dot):
                
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J
                #J_H_reg = J_H + 0.0001 * np.eye(J_H.shape[0])
                kp = 50.0

                # Define OSQP data
                P = sp.csc_matrix(np.eye(7))  # Quadratic term
                
                x_dot6 = np.array([1,0,0,0,0,0])
                self.gradient = self.dir_manipulability_gradient2(q)  # Linear term
                #print("info:")
                #print("q; ", q)
                #print("gradient; ", self.gradient)
                #print("µ; ", self.manipulability(q) )
                #print("q; ", q)
                # Define constraint matrix and bounds
                A = sp.csc_matrix(J_H)
                l = (DQd_dot.asVector() + kp*error).flatten()
                u = l  # Equality constraint

                # Create an OSQP object
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(P, -6*self.gradient, A, l, u, **self.osqp_settings)

                # Solve problem
                res = prob.solve()
                
                
                if res.info.status != 'solved':
                        print("The problem is ", res.info.status )                        
         
              
                # The problem was feasible (or possibly optimal)
                q_dot_ = res.x
                self.q_dot_last_ = q_dot_.copy()
                return q_dot_


        def quadratic_program_2(self, q, q_dot, DQd, DQd_dot):
                
                # try to give slack on orientation but that also affects position in DQ space
                x = self.forward_kinematics.forward_kinematics(q)
                
                error = (DQd - x).asVector()
                
                J = self.forward_kinematics.jacobian(q)
                
                J_H = 0.5*DQd.as_mat_right()@J

                kp = 50.0
                
                Omega = DQd_dot*DQd.inverse()*2.0
                v = Omega.as6Vector().flatten()

                
                # Define OSQP data
                P = sp.block_diag((1*np.eye(7), 100_000*np.eye(4)))  # Quadratic term
                x_dot6 = np.array([1,1,1,0,0,0])*0.3
                self.gradient = self.dir_manipulability_gradient2(q) 
                grad = np.hstack([-2*self.gradient, np.zeros(4)])           # Linear term
                #print(q)
                
                # Define constraint matrix and bounds
                slack_matrix = np.array([[1,0,0,0],
                                         [0,1,0,0],
                                         [0,0,1,0],
                                         [0,0,0,1],
                                         [0,0,0,0],
                                         [0,0,0,0],
                                         [0,0,0,0],
                                         [0,0,0,0]])
                
                slack_matrix2 = np.array([[0,0,0,0, 0,0,0, 1,0,0,0],
                                          [0,0,0,0, 0,0,0, 0,1,0,0],
                                          [0,0,0,0, 0,0,0, 0,0,1,0],
                                          [0,0,0,0, 0,0,0, 0,0,0,1]])
                
                test = np.hstack([J_H, slack_matrix])
                test2 = np.vstack([test, slack_matrix2])
                
                
                A = sp.csc_matrix(test2)
                l = np.concatenate([(DQd_dot.asVector() + kp*error).flatten(), -np.inf * np.ones(4)])
                u = np.concatenate([(DQd_dot.asVector() + kp*error).flatten(), np.inf * np.ones(4)])
                
                #print(l)
                # Create an OSQP object
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(P, grad, A, l, u, **self.osqp_settings)

                # Solve problem
                res = prob.solve()
                
                
                if res.info.status != 'solved':
                        print("The problem is infeasible.")
                        print(res.info.status)
                        # Handle the infeasible case (e.g., return None or raise an exception)
                        return None
              
                # The problem was feasible (or possibly optimal)
                q_dot_ = res.x[:7]
                
                return q_dot_

        def quadratic_program_4(self, q, q_dot, DQd, DQd_dot):
                
                # try to give slack on orientation
                x = self.forward_kinematics.forward_kinematics(q)
                
                x_error = DQd*x.inverse()

                pos_error = x_error.getPosition()
                
                o_error = x_error.real.log().getVector()

                Omega = DQd_dot*DQd.inverse()*2.0
                x_dot = Omega.as6Vector()

                error = np.array([*o_error.flatten(), *pos_error.flatten()])
                J = self.forward_kinematics.jacobian6(q)
                kp = 20.0

                # Define OSQP data
                P = sp.block_diag((0.01*np.eye(7), 1*np.eye(3)))  # Quadratic term
                q = np.zeros(10)                # Linear term

                # Define constraint matrix and bounds
                slack_matrix = np.array([[1,0,0],
                                         [0,1,0],
                                         [0,0,1],
                                         [0,0,0],
                                         [0,0,0],
                                         [0,0,0]])
                
                slack_matrix2 = np.array([[0,0,0,0, 0,0,0, 1,0,0],
                                          [0,0,0,0, 0,0,0, 0,1,0],
                                          [0,0,0,0, 0,0,0, 0,0,1]])
                
                test = np.hstack([J, slack_matrix])
                test2 = np.vstack([test, slack_matrix2])

                A = sp.csc_matrix(test2)
                l = np.concatenate([(x_dot.flatten() + kp*error), -np.inf * np.ones(3)])
                u = np.concatenate([(x_dot.flatten() + kp*error), np.inf * np.ones(3)])
                
                #print(l)
                # Create an OSQP object
                prob = osqp.OSQP()

                # Setup workspace and change settings
                prob.setup(P, q, A, l, u, **self.osqp_settings)

                # Solve problem
                res = prob.solve()
                
                
                if res.info.status != 'solved':
                        print("The problem is infeasible.")
                        print(res.info.status)
                        # Handle the infeasible case (e.g., return None or raise an exception)
                        return None
              
                # The problem was feasible (or possibly optimal)
                q_dot_ = res.x[:7]
                
                return q_dot_
               