import polyscope as ps
from pickle import load
import argparse
import numpy as np


# Import the necessary libraries
import os
import sys
from pathlib import Path

hanan_path = os.getenv('HANANLAB_PATH')
if not hanan_path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(hanan_path)

from geometry.utils import *
from utils.bsplines_functions import *

path = os.getcwd()
print(path)

# experiment dir
exp_dir = os.path.join(path, 'QS_project', 'experiments')

# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
file_name = name = parser.parse_args().file_name

file_name += '.pickle'


def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, file_name), 'rb') as f:
        data = load(f)
    return data


def main():
    """ Main function to visualize the data
    """
    # Load the data
    data = load_data()
    
    V = data['V'] # Vertices
    F = data['F'] # Faces
    l = data['l'] # Line congruence 
    n = data['n'] # Normal
    t1 = data['t1'] # Torsal direction 1
    t2 = data['t2'] # Torsal direction 2
    nt1 = data['nt1'] # Torsal normal 1
    nt2 = data['nt2'] # Torsal normal 2
    lt1 = data['lt1'] # Torsal l direction 1
    lt2 = data['lt2'] # Torsal l direction 2
    barycenters = data['bar'] # Barycenters
    lc = data['lc'] # Line congruence barycenters
    r_uv_surf = data['r_uv_surf'] # r_uv_surf
    
    l = l.reshape(-1, 3)
    l /= np.linalg.norm(l, axis=1)[:,None]
    n = n.reshape(-1, 3)

    # Working directory
    working_path = os.getcwd()


    # Remeshing data folder
    remeshing_dir = os.path.join(working_path, 'QS_project', 'data', 'Remeshing', name)

    # Frame Field remeshed obj
    remeshed_obj = os.path.join( remeshing_dir,  name+'_Remeshed.obj')

    # Read remeshed mesh
    ffV, ffF = read_obj(remeshed_obj)
   

    # Compute Torsal angles
    torsal_angles = np.arccos(np.abs(np.sum(nt1*nt2, axis=1)))*180/np.pi

    # Angle with normal
    ang_normal = np.arccos( np.sum( l*n, axis=1) )*180/np.pi

    # Get vertices
    v0, v1, v2, v3 = V[F[:,0]], V[F[:,1]], V[F[:,2]], V[F[:,3]]

    # Compute tangents
    du = v2 - v0
    dv = v1 - v3


    mean_diagonals = np.mean((np.linalg.norm(du, axis=1) + np.linalg.norm(dv, axis=1))/2)

    size_torsal = mean_diagonals/4

   
    planarity_opt = 0.5*(planarity_check(t1, lt1, lc) + planarity_check(t2, lt2, lc))

    ps.init()
    
    surf = ps.register_surface_mesh("Surface", V, F)

    # OPTIMIZED LC
    surf.add_vector_quantity("l", l.reshape(-1, 3), defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))

    surf.add_scalar_quantity("r_uv", r_uv_surf.flatten(), defined_on="vertices", enabled=True)

    # ANGLES WITH NORMAL SCALAR FIELD
    surf.add_scalar_quantity("Angles", ang_normal.flatten(), defined_on="vertices", enabled=True)

    surf.add_scalar_quantity("Torsal_Angles", torsal_angles, defined_on="faces", enabled=True)

    surf.add_scalar_quantity("Planarity", planarity_opt, defined_on="faces", enabled=True)

    torsal_dir_show(barycenters, t1, t2, size=size_torsal, rad=0.004)

    V_R = V + r_uv_surf.flatten()[:,None]*n.reshape(-1,3)

    ps.register_surface_mesh("C_uv", V_R, F)

    ps.register_surface_mesh("Remeshed", ffV, ffF)

    # Visualize the data
    
    ps.show()
    

main()


