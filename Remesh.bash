#source /home/anthony/anaconda3/etc/profile.d/conda.sh
#conda activate geo

#Exp_dir="/home/anthony/Hanan Lab/hananLab/QS_project/experiments/"
Exp_dir="/Users/cisneras/hanan/hananLab/QS_project/experiments/"
Exp_name="Nice_result.pickle"

file_path=$Exp_dir$Exp_name
echo $Exp_dir$Exp_name

python Quad_remesh/stress_remesher.py "$file_path" 1 0.5 300
