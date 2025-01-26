# Import the necessary libraries
import os
import sys
import time 
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


# Define Bsplines Surface directory
surface_dir = os.path.join(dir_path, "data", "Bsplines_Surfaces")
print("surface dir:", surface_dir)
experiment_dir = os.path.join(dir_path, "experiments")
reports_dir = os.path.join(dir_path, "data", "Reports")

# Optimization options ====================================================
# Create the parser
parser = argparse.ArgumentParser(description="Optimization")

# Argument for file name
parser.add_argument('file_name', type=str, help='File name to load')

# # U and V grid size
# parser.add_argument('deltaumin', type=float, help='delta value')
# parser.add_argument('deltaumax', type=float, help='delta value')
# parser.add_argument('deltavmin', type=float, help='delta value')
# parser.add_argument('deltavmax', type=float, help='delta value')

# Type of file : 1 pickle, 2 json
parser.add_argument('type', type=int, help='Read 1 pickle or 2 json file')

bspline_surf_name = parser.parse_args().file_name
dir =  1

class GUI:

    def __init__(self):
        self.run_opt     = False
        self.opt_type    = None # 0 LC opt; 1 Torsal opt
        self._iterations = 10
        self.opt         = Optimizer()
        self._it         = 0 
        self._recomp_it  = 25
        self.log         = []   
        self.time_opt    = 0.0
        self.name_saved  = bspline_surf_name

        # Optimization parameters
        # Sample grid
        self.bsp    =  None
        self.sample = [20, 20]
        self.tangle = 45
        self.angle  = 15
        self.step_size = 0.5

        # Visualization parameters
        self.u_range = [0.1,0.99]
        self.v_range = [0.1,0.99]
        self.u_pts   = None 
        self.v_pts   = None 
        

        self.weights = {
        "LC": 1, # Line congruence l.cu = 0, l.cv = 0
        "LC_Orth": 2, # Line congruence orthogonality with surface
        "Torsal": 1, # Torsal constraint
        "Torsal_Angle": 3, # Torsal angle constraint
        "Fairness": 0.1 # Fairness constraint
        }

    def export_data(self)->None:

        # Get RESULTS
        l, cp, tu1, tu2, tv1, tv2, nt1, nt2 = self.opt.uncurry_X("l", "rij", "u1", "u2", "v1", "v2", "nt1", "nt2")

        # Reshape Torsal normals
        nt1 = nt1.reshape(-1,3)
        nt2 = nt2.reshape(-1,3)

        # Update control points of r(u,v) spline surface
        self.vars["r_uv"][2] = cp  
        r_uv_surf = bisplev(self.u_pts, self.v_pts, self.vars["r_uv"])

        # Reshape Line congruence
        l = l.reshape(len(self.u_pts), len(self.v_pts), 3)
        l /= np.linalg.norm(l, axis=2)[:,:,None]

        # Angle with normal
        ang_normal = np.arccos( np.sum( l*self.vars["n"], axis=2) )*180/np.pi

        V, F =  Bspline_to_mesh(self.bsp, self.u_pts, self.v_pts)

        # Get vertices
        v0, v1, v2, v3 = V[F[:,0]], V[F[:,1]], V[F[:,2]], V[F[:,3]]

        # l (-1,3)
        l_f = l.reshape(-1,3)

        l0, l1, l2, l3 = l_f[F[:,0]], l_f[F[:,1]], l_f[F[:,2]], l_f[F[:,3]]

        lu = l2 - l0
        lv = l1 - l3

        # Compute tangents
        du = v2 - v0
        dv = v1 - v3

        # Compute barycenters
        barycenters = (v0 + v1 + v2 + v3)/4
        lc = (l0 + l1 + l2 + l3)/4

        # Get torsal directions
        t1 = unit(tu1[:,None]*du + tv1[:,None]*dv)
        t2 = unit(tu2[:,None]*du + tv2[:,None]*dv)

        lt1 = unit(tu1[:,None]*lu + tv1[:,None]*lv)
        lt2 = unit(tu2[:,None]*lu + tv2[:,None]*lv)




        V_R = V + r_uv_surf.flatten()[:,None]*self.vars["n"].reshape(-1,3)
        
        # Variable to save
        save_data = {
                    'surf': self.bsp,
                    'r_uv': self.vars["r_uv"],
                    'u_pts': self.u_pts,
                    'v_pts': self.v_pts,
                    'V': V,
                    'V_R': V_R,
                    'F': F,
                    'l': l,
                    'init_l': self.vars["init_l"],
                    'n': self.vars["n"],
                    't1': t1,
                    't2': t2,
                    'nt1': nt1,
                    'nt2': nt2,
                    'lt1': lt1,
                    'lt2': lt2,
                    'bar': barycenters,
                    'lc': lc,
                    'r_uv_surf': r_uv_surf
                      }
        
        save_file_path = os.path.join(experiment_dir, self.name_saved + ".pickle")

        # Save the variable to a file
        with open(save_file_path, 'wb') as file:
            pickle.dump(save_data, file)

        # Save report 
        ps.warning("Results saved in: " + save_file_path)


    def dump_to_log(self)->None:
        """ 
        Dump the log to a file information about the optimization
        """
        local_log = {
            "iterations": self.opt.it,
            "weights": self.weights,
            "time": self.time_opt,
            "u_range": self.u_range,
            "v_range": self.v_range,
            "Energies_Report": self.opt.final_enery_log_report()
        }
        self.log.append(local_log)

    def print_log(self)->None:
        """ 
        Print the log
        """
        print(self.log)
    
    def save_report(self)->None:
        """ 
        Save the report to a file
        """
        # Create the directory if it does not exist
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)

        # Save the report
        with open(os.path.join(reports_dir, bspline_surf_name+'_log.json'), 'w') as f:
            json.dump(self.log, f)

    def loop(self):

        # T_itle
        psim.TextUnformatted("Optimization Visualizer")
        psim.PushItemWidth(100)

     
        # Initial Visualization of the mesh
        _, self.u_range[0] = psim.InputFloat("U min", self.u_range[0])
        psim.SameLine()
        _, self.u_range[1] = psim.InputFloat("U max", self.u_range[1])
        psim.SameLine()
        _, self.sample[0] = psim.InputInt("U sample", self.sample[0])

        _, self.v_range[0] = psim.InputFloat("V min", self.v_range[0])
        psim.SameLine()
        _, self.v_range[1] = psim.InputFloat("V max", self.v_range[1])
        psim.SameLine()
        _, self.sample[1] = psim.InputInt("V sample", self.sample[1])


        
        if psim.Button("Draw Mesh"):
             # Get Grid points
            self.u_pts = np.linspace(self.u_range[0], self.u_range[1], self.sample[0])
            self.v_pts = np.linspace(self.v_range[0], self.v_range[1], self.sample[1])
            visualization_init(self.bsp, self.u_pts, self.v_pts)

        psim.Separator()


        _, self.angle = psim.InputFloat("Angle normal", self.angle)
        _, self.tangle = psim.InputFloat("Torsal Angle", self.tangle)

        psim.TextUnformatted("Line Congruence Optimization:")

        if psim.Button("Initialize Optimization"):
            self.opt_type = 0
            self.opt = Optimizer()
            self.vars = optimization_init(self.opt,
                                self.bsp,
                                self.sample,
                                self.u_range,
                                self.v_range,
                                self.step_size,
                                self.angle
                            )
            
        _, self.weights["LC"] = psim.InputFloat("Line Congruence Weight", self.weights["LC"])
        _, self.weights["LC_Orth"] = psim.InputFloat("Line Congruence Orthogonality Weight", self.weights["LC_Orth"])
            

        psim.Separator()

        psim.TextUnformatted("Torsal Field Optimization:")


        if psim.Button("Initialize Torsal Optimization"):
            self.opt_type = 1
            self.opt = optimization_torsal(self.opt,
                                self.sample,
                                self.bsp,
                                self.vars["r_uv"],
                                self.vars["n"],
                                self.u_pts,
                                self.v_pts,
                                self.step_size,
                                self.angle,
                                self.tangle
                            )
            
            
        _, self.weights["LC"] = psim.InputFloat("Line Congruence Weight T", self.weights["LC"])
        _, self.weights["LC_Orth"] = psim.InputFloat("Line Congruence Orthogonality Weight T", self.weights["LC_Orth"])
        _, self.weights["Torsal"] = psim.InputFloat("Torsal Weight", self.weights["Torsal"])
        _, self.weights["Torsal_Angle"] = psim.InputFloat("Torsal Angle Weight", self.weights["Torsal_Angle"])
        _, self.weights["Fairness"] = psim.InputFloat("Fairness Weight", self.weights["Fairness"])


        psim.Separator()

        # __iterations input
        _, self._iterations = psim.InputInt("Iterations", self._iterations)
        _, self._recomp_it = psim.InputInt("It per Recomp ", self._recomp_it)

        # Get the callbacks
        self.get_callbacks()


    def get_callbacks(self):


        # Run optimization button
        if psim.Button("Run Optimization"):
            self.run_opt = True

        if self._it >= self._iterations or self.opt.stop:
            
            self.dump_to_log()
            self.run_opt = False
            self._it = 0
            self.opt.stop = False
            #self.opt_type = None
            self.time_opt = 0.0

        if self.run_opt:
            self._it += 1
        
            
            if self.opt_type == 0:
                # Modify the mesh by the optimization
                it_time = time.time()
                optimization_gui(self.opt, self.weights)
                self.time_opt += time.time() - it_time
                # Update the visualization
                visualization_opt_lc(self.opt, self.bsp, self.vars["r_uv"], self.vars["n"], self.u_pts, self.v_pts)
            elif self.opt_type == 1:
                # Modify the mesh by the optimization
                it_time = time.time()
                if self._it%self._recomp_it == 0 and self.weights["Torsal_Angle"]!= 0:
                    self.opt.constraints["BS_Torsal"].recompute(self.opt.X, self.opt.var_idx)

                optimization_gui(self.opt, self.weights)
                self.time_opt += time.time() - it_time
                # Update the visualization
                visualization_opt_torsal( self.opt, self.bsp, self.vars['r_uv'], self.vars['n'], self.u_pts, self.v_pts)
        
        _, self.name_saved = psim.InputText("Name", self.name_saved)

        if psim.Button("Save Report"):
            self.print_log()
            self.save_report()

        psim.SameLine()

        if psim.Button("Export"):

            self.export_data()
            
    def start(self):

        # Load the B-spline surface
        if parser.parse_args().type == 1:
            # Load picke information
            def load_data():
                """ Function to load the data from a pickle file
                """
                with open(os.path.join(experiment_dir, bspline_surf_name+'_init.pickle'), 'rb') as f:
                    data = pickle.load(f)
                return data
            data = load_data()
            self.bsp0 = self.bsp = data["surf"]

        elif parser.parse_args().type == 2:
            self.bsp0 = self.bsp = get_spline_data(0, surface_dir, bspline_surf_name)

        self.u_pts = np.linspace(self.u_range[0], self.u_range[1], self.sample[0])
        self.v_pts = np.linspace(self.v_range[0], self.v_range[1], self.sample[1])

        # In_itialize polyscope
        ps.init()
        
        # In_itialize the visualization
        visualization_init(self.bsp, self.u_pts, self.v_pts)
        
        # Main loop
        ps.set_user_callback(self.loop)

        ps.show()


gui = GUI()
gui.start()