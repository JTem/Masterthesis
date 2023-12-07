import math
import numpy as np
import osqp
import scipy.sparse as sp
import scipy

class ModelPredictiveControl:
        def __init__(self, N, Nu, dof):
                
                self.N = N
                self.Nu = Nu
                self.dim_jac = 8
                self.dof = dof
                
                self.dim_vars = N*dof + Nu*(2*dof + self.dim_jac)
                
        
        def initializeHessian(self, Qx, Qu, Qs, Qdu):
                matrices = []
                for i in range(0, self.N):
                        matrices.append(Qx)
                for i in range(0, self.Nu):
                        matrices.append(Qu)
                for i in range(0, self.Nu):
                        matrices.append(Qs)
                for i in range(0, self.Nu):
                        matrices.append(Qdu)
                
                hessian = scipy.linalg.block_diag(*matrices)

                return sp.csc_matrix(hessian)
        
        def initializeGradient(self):
                gradient_dim = self.dof*(self.N + 2*self.Nu) + self.Nu*self.dim_jac

                return np.zeros(gradient_dim)
        
        def initializeConstraintVectors(self, Ts, x0, A, u0, joint_limits, velocity_limits, acceleration_limits):
                self.Ad0 = self.computeAd(A, Ts)
                dim_eqconstraint_vec = self.dof*(self.N + self.Nu) + self.Nu*self.dim_jac 
                dim_ineqconstraint_vec = self.dof*(self.N + 2*self.Nu)
                offset_u = self.dof*self.N + self.Nu*self.dim_jac
                
                eq_constraintVector = np.zeros(dim_eqconstraint_vec)
                eq_constraintVector[:self.dof] = -self.Ad0@x0
                eq_constraintVector[offset_u:offset_u + self.dof] = -u0
                
                ineq_constraintVector = np.zeros(dim_ineqconstraint_vec)
                for i in range(0, self.N):
                        ineq_constraintVector[(self.dof*i):(self.dof*i) + self.dof] = joint_limits
                        
                offset_u = self.dof*self.N
                offset_du = self.dof*(self.Nu + self.N)
                
                for i in range(0, self.Nu):
                        ineq_constraintVector[(offset_u + self.dof*i):(offset_u + self.dof*i) + self.dof] = velocity_limits                        
                        ineq_constraintVector[(offset_du + self.dof*i):(offset_du + self.dof*i) + self.dof] = acceleration_limits
                
                lower_constraint_vector = np.hstack([eq_constraintVector, -ineq_constraintVector])
                upper_constraint_vector = np.hstack([eq_constraintVector, ineq_constraintVector])
                
                return lower_constraint_vector, upper_constraint_vector
        
        
        def updateConstraintVector(self, lower_constraint_vector, upper_constraint_vector, x0, u0, ref_list):
                
                offset_ref = self.dof*self.N
                offset_u = self.dof*self.N + self.Nu*self.dim_jac
                
                dim_constraint_vec = self.dof*(self.N + self.Nu) + self.Nu*self.dim_jac 
                
                constraintVector = np.zeros(dim_constraint_vec)
                
                #print(lower_constraint_vector)
                #print(ref_list)
                for i in range(0, self.Nu):
                        lower_constraint_vector[(offset_ref + self.dim_jac*i):(offset_ref + self.dim_jac*i) + self.dim_jac] = ref_list[i]   
                        upper_constraint_vector[(offset_ref + self.dim_jac*i):(offset_ref + self.dim_jac*i) + self.dim_jac] = ref_list[i]
                         
                lower_constraint_vector[:self.dof] = -self.Ad0@x0
                lower_constraint_vector[offset_u:offset_u + self.dof] = -u0
                
                upper_constraint_vector[:self.dof] = -self.Ad0@x0
                upper_constraint_vector[offset_u:offset_u + self.dof] = -u0
                
                return lower_constraint_vector, upper_constraint_vector
                

        def updateConstraintMatrix(self, A_constraint, J_list):
                
                # Jacobian and slack
                offset_jac = self.dof*self.N
                
                for i in range(0, self.Nu):
                        A_constraint[(offset_jac + self.dim_jac*i):(offset_jac + self.dim_jac*i) + self.dim_jac, (offset_jac + self.dof*i):(offset_jac + self.dof*i) + self.dof] = J_list[i]
                
                return A_constraint
        
        
#         def updateConstraintMatrix2(self, csc_A_constraint, J_list):
                
#                 # Jacobian and slack
#                 offset_jac = self.dof*self.N
                
#                 for i in range(0, self.Nu):
#                         J = J_list[i]
#                         for j in range(0, self.dim_jac):
#                                 for k in range(0, self.dof):
#                                         csc_A_constraint[(offset_jac + self.dim_jac*i + j), (offset_jac + self.dof*i + k)] = J[j, k]
                
#                 return csc_A_constraint
        
        
#         def updateConstraintMatrix3(self, csc_A_constraint, J_list):
                
#                 # Jacobian and slack
#                 offset_jac = self.dof*self.N
                
#                 coo = csc_A_constraint.tocoo()
                
#                 for i in range(0, self.Nu):
#                         J = sp.coo_matrix(J_list[i])
#                         # Update the block
#                         for j, k, v in zip(J.row, J.col, J.data):
#                                 coo.data[(coo.row == offset_jac + self.dim_jac*i + j) & (coo.col == offset_jac + self.dim_jac*i + k)] = v

#                 # Convert back to CSC format
#                 updated_csc_matrix = coo.tocsc()
                
#                 return csc_A_constraint
        
        
        def initializeConstraintMatrix(self, A, B, Ts_list):
                
                EQ_dim1 = self.N*self.dof + self.Nu*(self.dim_jac + self.dof)
                matrix_dim2 = self.dof*(self.N + 2*self.Nu) + self.Nu*self.dim_jac
                INEQ_dim1 = self.N*self.dof + 2*self.Nu*self.dof
                A_constraint = np.zeros((EQ_dim1 + INEQ_dim1, matrix_dim2))
                
                for i in range(0, self.N):
                        
                        A_constraint[(self.dof*i):(self.dof*i) + self.dof, (self.dof*i):(self.dof*i) + self.dof] = -np.eye(self.dof)
                        
                        # add A
                        if(i > 0):
                                A_constraint[(self.dof*i):(self.dof*i) + self.dof, (self.dof*(i-1)):(self.dof*(i-1)) + self.dof] = self.computeAd(A, Ts_list[i+1])
                                
                        # for B
                        if i < self.Nu:
                                cntB2 = i*self.dof
                        
                        offsetB2 = self.dof*self.N
                        A_constraint[(self.dof*i):(self.dof*i) + self.dof, (offsetB2 + cntB2):(offsetB2 + cntB2) + self.dof] = self.computeBd(B, A, Ts_list[i])
                        
                
                # Jacobian and slack
                offset_jac = self.dof*self.N
                offset_Is = self.dof*(self.N + self.Nu)
                for i in range(0, self.Nu):
                        A_constraint[(offset_jac + self.dim_jac*i):(offset_jac + self.dim_jac*i) + self.dim_jac, (offset_jac + self.dof*i):(offset_jac + self.dof*i) + self.dof] = np.ones((self.dim_jac, self.dof))
                        A_constraint[(offset_jac + self.dim_jac*i):(offset_jac + self.dim_jac*i) + self.dim_jac, (offset_Is + self.dim_jac*i):(offset_Is + self.dim_jac*i) + self.dim_jac] = np.eye(self.dim_jac)
                      
                
                
                # u and du
                offset_du1 = self.dof*self.N + self.Nu*self.dim_jac
                offset_du2 = self.dof*self.N + self.Nu*(self.dof + self.dim_jac)
                offset_u2 = self.dof*self.N
                for i in range(0, self.Nu):
                        A_constraint[(offset_du1 + self.dof*i):(offset_du1 + self.dof*i) + self.dof, (offset_du2 + self.dof*i):(offset_du2 + self.dof*i) + self.dof] = np.eye(self.dof)*Ts_list[i]

                                
                        A_constraint[(offset_du1 + self.dof*i):(offset_du1 + self.dof*i) + self.dof, (offset_u2 + self.dof*i):(offset_u2 + self.dof*i) + self.dof] =  -np.eye(self.dof)
                                               
                        if(i > 0):
                                A_constraint[(offset_du1 + self.dof*i):(offset_du1 + self.dof*i) + self.dof, (offset_u2 + self.dof*(i-1)):(offset_u2 + self.dof*(i-1)) + self.dof] =  np.eye(self.dof)
                                
                
                #INEQ_Constraint
                for i in range(0, self.N + self.Nu):
                        A_constraint[(EQ_dim1 + self.dof*i):(EQ_dim1 + self.dof*i) + self.dof, (self.dof*i):(self.dof*i) + self.dof] = np.eye(self.dof)
                
                offset_du1 = self.dof*(self.N + self.Nu)
                offset_du2 = self.dof*(self.N + self.Nu) + self.Nu*self.dim_jac
                for i in range(0, self.Nu):
                        A_constraint[(EQ_dim1 + offset_du1 + self.dof*i):(EQ_dim1 + offset_du1 + self.dof*i) + self.dof, (offset_du2 + self.dof*i):(offset_du2 + self.dof*i) + self.dof] = np.eye(self.dof)
                        
                return A_constraint
                
        def computeDeltaTimeVector(self, Ts, N, Ts_lin_fact, Ts_quad_fact):
                dt_vec = []
                dt_vec.append(Ts)
                for i in range(0, N):
                        dt_vec.append(Ts + Ts_lin_fact*i*Ts + (Ts_quad_fact*i*Ts)**2)
                        
                return dt_vec
        
        def computeBd(self, B, A, Ts):
                Bd = np.zeros(B.shape)

                for i in range(1, 10):
                        Bd += (Ts**(i)*np.linalg.matrix_power(A, i-1)@B)/math.factorial(i)
    
                return Bd

        def computeAd(self, A, Ts):

                return scipy.linalg.expm(A*Ts)