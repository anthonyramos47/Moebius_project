# Optimizer class
import numpy as np
import time as tm
import pandas as pd
from optimization.Unit import Unit
from optimization.Step_Control import Step_Control
from geometry.utils import unit
from scipy.sparse import diags, vstack
from scipy.sparse.linalg import spsolve, gmres 
import matplotlib.pyplot as plt


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
        #self.J = None # Jacobian matrix
        #self.r = None # Residual vector
        self.b = None # Vector of residuals J.T@r
        self.X = None # Variable
        self.X0 = None # Initial variable 
        self.bestX = None # Best variable
        self.bestit = None # Best iteration
        self.prevdx = None # Previous dx
        self.H = None # Hessian matrix
        self.it = None # Iteration 
        self.step = None # Step size
        self.method = None # Method used to solve the problem
        self.energy = [] # Energy vector
        self.e_diff = 1 # Energy difference
        self.energy_dic = {} # Energy dictionary
        self.norm_energy_dic = {} # Energy normalized dictionary
        self.var_idx = {} # Variable indices
        self.var = 0 # Number of variables
        self.constraints = [] # List of constraints objects
        self.verbose = False # Verbose
        self.stop = False # Stop criteria
        self.fixed_values = False 
        self.fixed_idx = []

    def clear_energy(self):
        """
        Method to clear the energy
        """
        self.energy = []
        self.it = 0
        self.bestit = None
        self.bestX = None
        self.prevdx = None

    def clear_constraints(self):
        """
        Method to clear the constraints
        """
        self.constraints = []

    def reset_it_optimizer(self):
        """
        Method to reset the optimizer
        """
        self.clear_energy()
        self.clear_constraints()
        #self.X = self.X0.copy()
        self.prevdx = None
        self.it = 0
        self.bestX = None
        self.bestit = None
    

    def report_energy(self, name="Final_Energy_plot"):
        # Save energy per constraint to a file
        with open(name+"_energy_per_constraint.data", "w") as file:
            file.write(f"ENERGY REPORT\n")
            file.write("===========================================\n")
            for en_name, energy in self.energy_dic.items():
                file.write(f"{en_name}: {energy}\n")
            file.write("===========================================\n")
            file.write("=============Final Energy ==================\n")

            file.write(f"Final Energy: {self.energy[-1]}\n")
            file.write(f"Best iteration: {self.bestit + 1}\nBest energy: {self.energy[self.bestit]}")



        print(f"Final Energy: {self.energy[-1]}")
        plot = plt.plot(self.energy)
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        plt.title('Energy per iteration')
        plt.xlim(0, len(self.energy))
        plt.grid()
        # Put point markers on the plot
        plt.scatter(range(len(self.energy)), self.energy, color='r')
        plt.savefig(name)
    
    def get_energy_per_constraint(self):
        print(f"ENERGY REPORT\n")
        print("===========================================\n")
        for name, energy in self.energy_dic.items():
            print(f"{name}: {energy}")
        print("===========================================\n")
        print("=============Final Energy ==================\n")
        print(f"Final Energy: {self.energy[-1]}\n")
        print(f"Best iteration: {self.bestit + 1}\nBest energy: {self.energy[self.bestit]}\n\n")

    def get_norm_energy_per_constraint(self):
        print(f"Normalized ENERGY REPORT\n")
        print("===========================================\n")
        for name, energy in self.norm_energy_dic.items():
            print(f"{name}: {energy}")
        print("===========================================\n")
        print("=============Final Energy ==================\n")
        print(f"Final Iteration: {self.it} Energy: {sum(e_i for e_i in self.norm_energy_dic.values())}\n")
        

    def add_variable(self, var_name, dim) -> None:
        """
            Method to add a variable to the optimizer
            Input:
                var_name: Name of the variable
                dim: Dimension of the variable
        """
        self.var_idx[var_name] = np.arange(self.var, self.var + dim)
        self.var += dim

    def init_variables(self, X) -> None:
        """
            Method to set the variables of the optimizer
            Input:
                X: Variables
        """
        self.X = X
        self.X0 = X.copy()

    def init_variable(self, name, vals):
        """
            Method to set the value of a variable
            Input:
                name: Name of the variable
                vals: Value of the variable
        """
        self.X[self.var_idx[name]] = vals
        self.X0[self.var_idx[name]] = vals

    def add_constraint(self, constraint, args, w=1, ce=0) -> None:
        """
            Method to add a constraint to the optimizer
            Input:
                constraint: Constraint class
                w: Weight of the constraint
                args: arguments of the constraint
        """

        constraint.sum_energy = ce
        # Add constraint to the optimizer
        constraint._initialize_constraint(self.X, self.var_idx, *args)
        constraint.set_weigth(w)

        self.constraints.append(constraint)

    def control_var(self, var_name, w) -> None:
        """
            Method to add a unit constraint to the optimizer. The unit constraint is a constraint that forces the variable to be unit.
            Input:
                var_name: Name of the variable
                var_indices: Indices of the variable
                dim: Dimension of the variable
        """
        # Initialize constraint
        SC = Step_Control()
        #unit.initialize_constraint(self.X, self.var_idx, var_name, dim)
        SC._initialize_constraint(self.X, self.var_idx, var_name)
        SC.w = w
        SC.name = var_name + "_step_control"
        #self.energy_vector = np.zeros(len(self.X))
        self.constraints.append(SC)
        # Add constraint
        #self.get_gradients(unit)



    def unitize_variable(self, var_name, dim, w) -> None:
        """
            Method to add a unit constraint to the optimizer. The unit constraint is a constraint that forces the variable to be unit.
            Input:
                var_name: Name of the variable
                var_indices: Indices of the variable
                dim: Dimension of the variable
        """
        # Initialize constraint
        unit = Unit()
        #unit.initialize_constraint(self.X, self.var_idx, var_name, dim)
        unit._initialize_constraint(self.X, self.var_idx, var_name, dim)
        unit.w = w
        unit.name = var_name + "_unit"
        #self.energy_vector = np.zeros(len(self.X))
        self.constraints.append(unit)
        # Add constraint
        #self.get_gradients(unit)

    def fix_variable(self, var, idx) -> None:
        """
            **-- Warning: Method not ready yet --**
            Method to fix vertices of the mesh. Not used in the current implementation
        """
        self.fixed_values = True 
        self.fixed_idx.append(self.var_idx[var][idx])


    def fix_values_H(self):
        """
            Method to fix the values of the variables in the Hessian matrix
        """
        if self.fixed_values:
            for idxs  in self.fixed_idx:
                self.H[idxs, :] = 0
                self.H[:, idxs] = 0
                #self.H[idxs, idxs] = 1


    def initialize_optimizer(self, method= "LM", step = 0.8, print=0) -> None:
        """
        Initialize the optimizer( variables, step size)
        """
        # Initialize variables
        self.X = np.zeros(self.var)
        self.X0 = np.zeros(self.var)
        self.it = 0
        self.step = step
        self.method = method
        self.verbose = print


    def get_gradients(self) -> None:
        """ Add constraint to the optimizer
            Input:
                constraint: Constraint class
                args: arguments of the constraint
        """
        stacked_H = []
        stacked_b = []

        #total = 0
        for constraint in self.constraints:
            
            # Add J, r to the optimizer
            if constraint.w != 0:
                
                #initial_time = tm.time()
                # Compute J, r for the constraint
                constraint._compute(self.X, self.var_idx)
                #final_time = tm.time()
                # total += final_time - initial_time
                #print(f"Time to compute {constraint.name}: {final_time - initial_time}")

                #initial_time = tm.time()
                # Add J, r to the optimizer                
                #stacked_J.append(np.sqrt(constraint.w) * constraint._J)
                stacked_H.append( constraint.w * constraint._J.T.dot(constraint._J))
                stacked_b.append( constraint.w * constraint._J.T.dot(constraint._r))    
                final_time = tm.time()
                # # total += final_time - initial_time
                #print(f"Time to stack and Hess {constraint.name}: {final_time - initial_time}\n\n")

                # Add energy to the energy dictionary
                if constraint.name is not None and constraint.sum_energy:
                        self.energy_dic[constraint.name] = constraint.w * np.sum(constraint._r**2)
                        self.norm_energy_dic[constraint.name] = constraint.w * np.mean(constraint._r**2)
        #print(f"\nTotal time to compute constraints: {total}")
        
        if len(stacked_H) == 1:
            self.H = stacked_H[0]
            self.b = stacked_b[0]
        else:
            #initial_time = tm.time()
            self.H = sum(H_i for H_i in stacked_H)
            self.b = sum(b_i for b_i in stacked_b)
            #final_time = tm.time()

            #print(f"Time to sum Hessians: {final_time - initial_time}")

    def optimize_step(self):
        if self.method == 'LM': # Levenberg-Marquardt
            self.LM()
        elif self.method == 'PG': # Projected Gauss-Newton
            self.PG()
        else:
            print("Error: Solver not implemented or not specified")
    
    def optimize(self, it=1):
        while self.it < it and not self.stop:
            self.get_gradients()
            self.optimize_step()

            self.stop_criteria()

    def stop_criteria(self):
        if (self.prevdx > 1e-8 or self.prevdx is None) and self.it < 5000:
            self.stop = False
        else:
            self.stop = True
        


    def LM(self):
        # Levenberg-Marquardt method for non-linear least squares
        # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        # Solve for (J^TJ + lambda*I) dx = -J^Tr, lambda = max(diag(J^TJ))*1e-8

        # Get J 
        #J = self.J

        # Compute pseudo Hessian
        #H = (J.T * J).tocsc()
        
        # # Calculate the value to add to the diagonal
        # add_value = self.H.max() * 1e-8

        # # Create a diagonal matrix with the values to add
        # diagonal_values = np.array([add_value] * self.H.shape[0])
        # diagonal_matrix = diags(diagonal_values, 0, format='csc')

        # # Add the diagonal_matrix to H
        # self.H = self.H + diagonal_matrix
        #self.fix_values_H()

        #mu = 1e-3 # Regularization parameter
        mu = self.H.max() * 1e-8
        

        #initial_time = tm.time()
        # Extract the diagonal elements from the matrix H_csr
        diagonal = self.H.diagonal()

        # Create a diagonal matrix where each diagonal entry is the reciprocal of the original matrix's diagonal
        #M = diags(1.0 / diagonal)
        
        self.H.setdiag(diagonal + mu)

        dx = spsolve(self.H, -self.b)

        # # Solve the linear system
        # dx, exitCode = gmres(self.H, -self.b, M=M)
        # if exitCode != 0:
        #     print(f"Solver did not converge at iteration, exit code: {exitCode}")
    
        #final_time = tm.time()
        #print(f"Time to solve linear system: {final_time - initial_time}")
        
        #b = -J.T@self.r
        
        # Store previous dx norm
        self.prevdx = np.linalg.norm(self.step*dx)

        energy = sum(e_i for e_i in self.energy_dic.values())

        energy_norm = sum(e_i for e_i in self.norm_energy_dic.values())   

        if len(self.energy) > 1:
            self.e_diff = abs(energy - self.energy[-1])
            
        # Append energy
        self.energy.append(energy)

        # Update variables
        self.update_variables(dx)

        # Store best X and iteration
        if self.it == 0:
            self.bestX = self.X
            self.bestit = self.it
        else:
            if energy < self.energy[self.bestit]:
                self.bestX = self.X
                self.bestit = self.it
        
        # Update iteration
        self.it +=1
        
        if self.verbose:
            # Print energy
            print(f" E {self.it}: {energy}\t {energy_norm}\t dx: {self.prevdx}")


    def get_variables(self):
        # Return variables
        print(f"Best iteration: {self.bestit + 1}\t Best energy: {self.energy[self.bestit]}")
        return self.bestX
        
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

        # if self.it%10 ==0:
        #     self.step *= 0.9
        
        if self.method == "LM":
            self.X += self.step*arg
        elif self.method == "PG":
            pass
    
    def uncurry_X(self, *v_idx):
        """ Function to uncurry the variables
            Input:
                v_idx: Variable indices
        """

        if len(v_idx) == 1:
            return self.X[self.var_idx[v_idx[0]]]
        else:
            return [self.X[self.var_idx[k]] for k in v_idx]

    def reset(self):
        """ Function that resets the optimizer to the initial state.
        """

        self.X = self.X0.copy()
        self.prevdx = None
        self.energy = []
        self.it = 0
        self.bestX = None
        self.bestit = None
        self.clear_constraints()


    def force_unit_variable(self, v_name, dim):
        """ Function that forces the variables to be unit vectors.
            Input:
                v_name: Name of the variable
                dim: Dimension of the variable
        """
        self.X[self.var_idx[v_name]] = unit(self.X[self.var_idx[v_name]].reshape(-1, dim)).flatten()

