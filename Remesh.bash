#Windows source /home/anthony/anaconda3/etc/profile.d/conda.sh
# Mac
source /Users/cisneras/opt/anaconda3/etc/profile.d/conda.sh

#Exp_dir="/home/anthony/Hanan Lab/hananLab/QS_project/experiments/"
Exp_dir="/Users/cisneras/hanan/hananLab/QS_project/experiments/"

#Exp_name="Nice_result.pickle"
Exp_name=$1

Pickle_name=$Exp_name".pickle"

conda activate hananJ

python QS_project/triangulation.py $1

conda deactivate
conda activate geo


file_path=$Exp_dir$Exp_name
echo $Exp_dir$Exp_name

python Quad_remesh/stress_remesher.py "$file_path" 0.5 0.01 300

conda deactivate
conda activate hananJ

python QS_project/cross_field_sep.py $Exp_name