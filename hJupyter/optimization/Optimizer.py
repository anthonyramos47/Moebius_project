# Optimizer class
import numpy as np
import time as tm
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg


class Optimizer():
    def __init__(self) -> None:
        self.J = None # Jacobian matrix
        self.r = None # Residual vector
        self.X = None # Variable
        self.H = None # Hessian matrix
        self.it = 0
        self.step = 0.8 
        self.energy = [] # Energy vector

    def fix_vertices(self, fix_vertices) -> None:
        # The structure of our Jacobian is J = [ V | aux variables]
        # If we want to fix a vertex, we need to set the corresponding column of the Jacobian to zero

        self.J[:, fix_vertices*3    ] = 0
        self.J[:, fix_vertices*3 + 1] = 0
        self.J[:, fix_vertices*3 + 2] = 0



    def initialize_optimizer(self, X) -> None:
        # Initialize variables
        self.X = X

    def add_constraint(self, constraint) -> None:
        ## Add constraint to the dictionary
        # Input:
        #   name: name of the constraint
        #   constraint: Constraint class
        #  mesh: mesh class
        #  X: variables

        # Add Jacobian and residual
        if self.J is None:
            self.J =  np.sqrt(constraint.w) * constraint.J
            # Get number of rows of the Jacobian
            self.r =  np.sqrt(constraint.w) * constraint.r
        else:
            self.J = np.vstack((self.J, np.sqrt(constraint.w) * constraint.J))
            self.r = np.concatenate((self.r, np.sqrt(constraint.w) * constraint.r))
 

    def optimize(self, name_solv):
        if name_solv == 'LM': # Levenberg-Marquardt
            sol = self.LM()
        elif name_solv == 'PG': # Projected Gauss-Newton
            sol = self.PG()
        else:
            print("Error: Solver not implemented")
            sol = -1 
        return sol

    def LM(self):
        # Levenberg-Marquardt method for non-linear least squares
        # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        # Solve for J^TJ + lambda*I dx = -J^Tr, lambda = max(diag(J^TJ))*1e-6

        
        # Compute pseudo Hessian
        H = self.J.T@self.J
        H[np.diag_indices_from(H)] += np.diag(H).max()*1e-6

        # Sparse matrix H
        H = csc_matrix(H)
    
        # Solve for dx
        dx = linalg.spsolve(H, -self.J.T@self.r)

        # Update r
        # self.r = self.r + self.J@dx
        # Compute energy
        energy = self.r.T@self.r

        print(f" E: {energy}")
        self.energy.append(energy)

        # Update variables
        self.update_variables("LM", dx)

        self.it +=1

        # Clear constraints
        self.clear_constraints()
        
    def PG(self):
        # To be implemented
        pass 
    
    def log_simple(self, filename) -> None:
        # Create a log file with the total energy of the system

        # Open file
        log_file = open(filename, "w")

        # Write energy
        for i in range(len(self.energy)):
            log_file.write(f"{i} {self.energy[i]}\n")
        
        log_file.close()


    def print_log(self) -> None:

        # Print energy
        df = pd.DataFrame(self.energy)

        df.columns = ["Energy"]
        
        print(df)
    
    def update_variables(self, name, arg) -> None:
        # Update variables

        # if self.it % 5 == 0:
        #     self.step *= 0.8

        if name == "LM":
            self.X += self.step*arg
        elif name == "PG":
            pass
        

    def clear_constraints(self):
        # Clear Jacobian and residual
        self.J = None
        self.r = None



