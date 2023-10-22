# Optimizer class
import numpy as np
import time as tm
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, linalg


class Optimizer():
    def __init__(self) -> None:
        """
            The optimizer is a class that allows to solve non-linear least squares problems
            using two different methods: Levenberg-Marquardt and Projected Gauss-Newton, of the form
                J^TJ dx = -J^Tr (LM) or J^TJ (X + l I) = J^T.X - r dx (PG)

            The optimizer is initialized with the variables to optimize (X) and the step size.

            The optimizer is then updated with constraints. Each constraint is a class that inherits from the Constraint class.
            The constraints are added to the optimizer using the add_constraint(constraint, *args) method. 
            The arguments of the constraint are passed as *args.

            The optimizer is then solved using the optimize(name_solv) method. The name_solv parameter is a string that can be
            either "LM" for Levenberg-Marquardt or "PG" for Projected Gauss-Newton.
        """
        self.J = None # Jacobian matrix
        self.r = None # Residual vector
        self.X = None # Variable
        self.H = None # Hessian matrix
        self.it = None # Iteration 
        self.step = None # Step size
        self.method = None # Method used to solve the problem
        self.energy = [] # Energy vector

    def fix_vertices(self, fix_vertices) -> None:
        """
            **-- Warning: Method not ready yet --**
            Method to fix vertices of the mesh. Not used in the current implementation
        """

        # The structure of our Jacobian is J = [ V | aux variables]
        # If we want to fix a vertex, we need to set the corresponding column of the Jacobian to zero
        # self.J[:, fix_vertices*3    ] = 0
        # self.J[:, fix_vertices*3 + 1] = 0
        # self.J[:, fix_vertices*3 + 2] = 0
        pass



    def initialize_optimizer(self, X, method= "LM", step = 0.8) -> None:
        """
        Initialize the optimizer( variables, step size)
        """
        # Initialize variables
        self.X = X
        self.it = 0
        self.step = step
        self.method = method

    def add_constraint(self, constraint, *args) -> None:
        """ Add constraint to the optimizer
            Input:
                constraint: Constraint class
                args: arguments of the constraint
        """
        
        # Add J, r to the optimizer
        if constraint.w != 0:
            
            # Compute J, r for the constraint
            constraint.compute(self.X, *args)

            # Add J, r to the optimizer
            if self.J is None:
                self.J =  np.sqrt(constraint.w) * constraint.J
                self.r =  np.sqrt(constraint.w) * constraint.r
            else:
                self.J = np.vstack((self.J, np.sqrt(constraint.w) * constraint.J))
                self.r = np.concatenate((self.r, np.sqrt(constraint.w) * constraint.r))
        

 

    def optimize(self):
        

        if self.method == 'LM': # Levenberg-Marquardt
            sol = self.LM()
        elif self.method == 'PG': # Projected Gauss-Newton
            sol = self.PG()
        else:
            print("Error: Solver not implemented or not specified")
            sol = -1 
        return sol

    def LM(self):
        # Levenberg-Marquardt method for non-linear least squares
        # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        # Solve for (J^TJ + lambda*I) dx = -J^Tr, lambda = max(diag(J^TJ))*1e-6

    
        # Compute pseudo Hessian
        H = self.J.T@self.J
        
        H[np.diag_indices_from(H)] += np.diag(H).max()*1e-8

        # Sparse matrix H
        H = csc_matrix(H)
        
        # Solve for dx
        dx = linalg.spsolve(H, -self.J.T@self.r)

        # Update r
        # self.r = self.r + self.J@dx

        # Compute energy
        energy = self.r.T@self.r

        # Append energy
        self.energy.append(energy)

        # Update variables
        self.update_variables(dx)

        # Update iteration
        self.it +=1
        
        # Print energy
        print(f" E {self.it}: {energy}")

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
    
    def update_variables(self, arg) -> None:
        # Update variables
        
        if self.method == "LM":
            self.X += self.step*arg
        elif self.method == "PG":
            pass
        

    def clear_constraints(self):
        # Clear Jacobian and residual
        self.J = None
        self.r = None



