import numpy as np
from Simulation.ForwardKinematics import ForwardKinematics
from neura_dual_quaternions import DualQuaternion

class CommandExecutor:
        
        def __init__(self, task_list, init_position, num_loops, dof):
                self.task_list = task_list
                self.task_counter = 0
                self.current_position = init_position
                self.current_velocity = np.zeros(dof)
                self.current_acceleration = np.zeros(dof)
                self.done = False
                self.num_loops = num_loops

                self.dof = dof
                self.forward_kinematics = ForwardKinematics()

                self.current_task = self.task_list[self.task_counter]
                self.num_tasks = len(self.task_list)
                self.current_task.setStartPosition(self.current_position)

                self.current_cartesian_target = DualQuaternion.basicConstructor(1,0,0,0, 0,0,0,0)
        
    
        def run(self, dt):
                self.current_task.run(dt)

                if self.current_task.done:
                        if self.task_counter < self.num_tasks - 1:
                                self.task_counter+=1
                                #print("go if")
                                
                        elif self.num_loops-1 > 0:
                                #print("go elif!")
                                self.task_counter = 0
                                self.num_loops -= 1
                                for task in self.task_list:
                                        task.reset()
                                        
                        else:
                                self.done = True
                                print("Done!")


                        self.current_task = self.task_list[self.task_counter]
                        self.current_task.setStartPosition(self.current_position)
                        self.current_task.setStartCartPosition(self.forward_kinematics.forward_kinematics(self.current_position))
                        self.current_velocity = np.zeros(self.dof)
                        self.current_acceleration = np.zeros(self.dof)

                self.updatePosition()
                self.updateVelocity()
                #self.updateAcceleration()
                self.updateCartesianTarget()
        
        def updatePosition(self):
                self.current_position = self.current_task.getPosition()

        def updateVelocity(self):
                self.current_velocity = self.current_task.getVelocity()

        def updateAcceleration(self):
                self.current_acceleration = self.current_task.getAcceleration()

        def updateCartesianTarget(self):
                self.current_cartesian_target = self.current_task.getCartesianTarget()
                
        def getPosition(self):
                return self.current_position

        def getCartesianTarget(self):
                return self.current_cartesian_target