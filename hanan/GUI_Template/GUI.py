# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
# Linux 
path = os.getenv('HANANLAB_PATH')
if not path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(path)

# Verification of the path
print(path)

# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp


##==Import the necessary classes and functions from the hananLab/hanan directory==##

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Local files
from optimization_functions import *
from visualization_functions import *

# Optimization classes
from optimization.Optimizer import Optimizer

# Directory where the data is stored ======================================

# Here you can add the directory where you want to save the results
dir_path = os.getcwd()
print(dir_path)


class GUI:

    def __init__(self):
        # GUI Variables
        self.run_opt     = False 
        self._iterations = 10 
        self.opt         = Optimizer()
        self._it         = 0 
        self.log         = []
        self.time_opt    = 0.0


        # Here can be be added 
        # Optimization and Vis parameters
        
        # Dictionary of weights used
        self.weights     = {}  

    def export_data(self)->None:
        """
        Function to export data after or during optimization
        """
        pass

    def dump_to_log(self)->None:
        """ 
        Dump the log to a file information about the optimization
        """

        # Sample of log
        #   local_log = {
        #     "iterations": self.opt.it,
        #     "weights": self.weights,
        #     "time": self.time_opt,
        #     "Energies_Report": self.opt.final_enery_log_report()
        # }
        pass 

    def print_log(self)->None:
        """ 
        Print the log
        """
        # print(self.log)
        pass

    def loop(self):

        # T_itle
        psim.TextUnformatted("Optimization Visualizer")

        # __iterations input
        _, self._iterations = psim.InputInt("Iterations: ", self._iterations)

        # Get the callbacks
        self.get_callbacks()
    

    def get_callbacks(self):

        # Run optimization button
        if psim.Button("Run Optimization"):
            self.run_opt = True

        if self._it == self._iterations:
            self.run_opt = False
            self._it = 0

        if self.run_opt:
            
            # Modify the mesh by the optimization
            optimization_gui(self.opt)
            # Update the visualization
            visualization()


            self._it += 1

    def start(self):

        # In_itialize polyscope
        ps.init()
        
        # In_itialize the visualization
        visualization_init()
        
        # Main loop
        ps.set_user_callback(self.loop)

        ps.show()


gui = GUI()
gui.start()