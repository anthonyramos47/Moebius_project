source /home/anthony/anaconda3/etc/profile.d/conda.sh
conda activate geo

Exp_dir="/home/anthony/Hanan Lab/hananLab/QS_project/experiments/"
Exp_name="Florian_A_20_TA_60.pickle"

file_path=$Exp_dir$Exp_name
echo $Exp_dir+$Exp_name

python Quad_remesh/stress_remesher.py "$file_path" 4 1 300
