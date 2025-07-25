#PBS -N GNN_MIDS
#PBS -q gpu
#PBS -o /lustre/home/mkrizman/MIDS-GNN/HPC/jobs/output/
#PBS -e /lustre/home/mkrizman/MIDS-GNN/HPC/jobs/error/
#PBS -M marko.krizmancic@fer.hr
#PBS -m ae
#PBS -l select=1:ncpus=16:ngpus=1:mem=32GB -l walltime=72:00:00

export http_proxy="http://10.150.1.1:3128"
export https_proxy="http://10.150.1.1:3128"

cd $HOME/MIDS-GNN

apptainer run --nv $HOME/Topocon_GNN/gnn_fiedler_approx/HPC/gnn_fiedler.sif wandb agent $sweep_ID
