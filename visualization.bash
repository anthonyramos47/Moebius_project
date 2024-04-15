#source /home/anthony/anaconda3/etc/profile.d/conda.sh

name=$1

#conda activate hananJ
python QS_project/experiments_vis.py parameters.json /Users/cisneras/hanan/hananLab/hanan/approximation/experiments/$name

# conda activate geo
# python Quad_remesh/stress_remesher.py parameters.json TriMesh
