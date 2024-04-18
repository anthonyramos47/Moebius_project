#Windows source /home/anthony/anaconda3/etc/profile.d/conda.sh
# Mac
source /Users/cisneras/opt/anaconda3/etc/profile.d/conda.sh

#Exp_dir="/home/anthony/Hanan Lab/hananLab/QS_project/experiments/"
Exp_dir="/Users/cisneras/hanan/hananLab/QS_project/experiments/"

#Exp_name="Nice_result.pickle"
Exp_name=$1

conda activate hananJ

cd QS_project

python back_mapper.py $1
python foot_point_bspline.py $1
