# Optimizer class
import numpy as np
import time as tm
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg


class Optimizer():
    def __init__(self) -> None:
        self.J = None # Jacobian matrix
        self.r = None # Residual vector
        self.x = None # Solution vector
        self.H = None # Hessian matrix
        self.energy = [] # Energy vector

    def add_constraint(self, name, constraint, mesh) -> None:
        ## Add constraint to the dictionary
        # Input:
        #   name: name of the constraint
        #   constraint: Constraint class

        # Compute constraint and get Jacobian and residual
        constraint.compute(mesh)
        J, r = constraint.J, constraint.r


        # Add Jacobian and residual
        if self.J is None:
            self.J = J
            # Get number of rows of the Jacobian
            self.r = r
        else:
            self.J = np.vstack((self.J, J))
            self.r = np.vstack((self.r, r))
 

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
        self.energy.append(energy)

        return dx
        
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

    def log_complete(self, filename) -> None:
        # Create a log file with the energy of each constraint

        # create data frame with dictionary of constraints
        df = pd.DataFrame.from_dict(self.constraints)

        # Write to csv
        df.to_csv(filename, index=False)


    def clear_constraints(self):
        # Clear Jacobian and residual
        self.J = None
        self.r = None



