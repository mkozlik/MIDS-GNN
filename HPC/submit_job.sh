#PBS -N GNN_MIDS
#PBS -q login-gpu
#PBS -o /lustre/home/mkrizman/MIDS-GNN/HPC/jobs/output/
#PBS -e /lustre/home/mkrizman/MIDS-GNN/HPC/jobs/error/
#PBS -l select=1:ncpus=4:ngpus=1:mem=16GB

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

cd $HOME/MIDS-GNN

apptainer run --nv HPC/gnn_fiedler.sif MIDS_script.py --standalone --eval-type basic --eval-target best
