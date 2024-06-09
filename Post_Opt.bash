#Windows source /home/anthony/anaconda3/etc/profile.d/conda.sh
# Mac
source /Users/cisneras/opt/anaconda3/etc/profile.d/conda.sh

#git pull origin LastC

Exp_name=$1

conda activate hananJ

cd QS_project

python foot_point_bspline.py $Exp_name 1

python Post_optimization_GUI.py $Exp_name
