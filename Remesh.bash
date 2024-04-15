#Windows source /home/anthony/anaconda3/etc/profile.d/conda.sh
# Mac
source /Users/cisneras/opt/anaconda3/etc/profile.d/conda.sh

conda activate geo

#Exp_dir="/home/anthony/Hanan Lab/hananLab/QS_project/experiments/"
Exp_dir="/Users/cisneras/hanan/hananLab/QS_project/experiments/"

#Exp_name="Nice_result.pickle"
Exp_name="Complex_res"

Pickle_name=$Exp_name".pickle"

file_path=$Exp_dir$Exp_name
echo $Exp_dir$Exp_name

python Quad_remesh/stress_remesher.py "$file_path" 0.1 0.001 300

conda deactivate
conda activate hananJ

python QS_project/cross_field_sep.py $Exp_name