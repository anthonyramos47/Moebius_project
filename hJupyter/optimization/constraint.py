# Creation of the constraint class
# This is meant to be a template to compute constraints
# The main idea is to pass a mesh and return a residual and a Jacobian
import numpy as np
import geometry as geo


class Constraint():
    def __init__(self) -> None:
        self.X = None # Variables
        self.J = None # Jacobian matrix
        self.J0 = None # Constant Jacobian
        self.r = None # Residual vector
        self.const = None # Num Constraints
        self.var = None # Num Variables

    def initialize_constraint(self, mesh) -> None:
        """
        Method to initialize the all initial data for the constraint
        """
        pass

    def compute(self, mesh) -> None:
        pass

    



        
