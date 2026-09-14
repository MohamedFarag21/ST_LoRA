#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run all four visualisation jobs
# Usage: bash run_all_viz.sh
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="${ST_LORA_ROOT}/code"
FIG_DIR="${ST_LORA_ROOT}/results/lora_paper/figures"
mkdir -p "${FIG_DIR}/logs"

# ── 1. Full FT selective prediction BG/FG viz ────────────────────────────────
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=viz_fullft_sel
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --gres=gpu:1 --time=1:00:00

module purge; module load CUDA/12.6.0; module load Miniforge3; source activate ssl
python ${SCRIPT_DIR}/selective_pred_viz_fullft.py \
    --seed 1337 --shot_ids 2 3 4 5 \
    --batch_size 4 --num_workers 8 --gpu 0 --height 1280 --width 720
EOF

# ── 2. MC Dropout selective prediction BG/FG viz ─────────────────────────────
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=viz_mcdrop_sel
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --gres=gpu:1 --time=1:00:00

module purge; module load CUDA/12.6.0; module load Miniforge3; source activate ssl
python ${SCRIPT_DIR}/selective_pred_viz_mcdropout.py \
    --seed 1337 --T 4 --dropout_p 0.5 \
    --batch_size 4 --num_workers 8 --gpu 0 --height 1280 --width 720
EOF

# ── 3. AURC comparison bar chart ─────────────────────────────────────────────
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=viz_compare_aurc
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_devel
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=8G --gres=gpu:0 --time=0:15:00

module purge; module load CUDA/12.6.0; module load Miniforge3; source activate ssl
python ${SCRIPT_DIR}/compare_aurc.py --out_dir ${FIG_DIR}
EOF

# ── 4. Reliability diagrams ───────────────────────────────────────────────────
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=viz_reliability
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --gres=gpu:1 --time=2:00:00

module purge; module load CUDA/12.6.0; module load Miniforge3; source activate ssl
python ${SCRIPT_DIR}/compare_reliability.py \
    --seed 1337 --n_bins 15 --T 4 \
    --batch_size 4 --num_workers 8 --gpu 0 --height 1280 --width 720
EOF

# ── 5. Uncertainty maps ───────────────────────────────────────────────────────
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=viz_unc_maps
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G --gres=gpu:1 --time=1:00:00

module purge; module load CUDA/12.6.0; module load Miniforge3; source activate ssl
python ${SCRIPT_DIR}/compare_uncertainty_maps.py \
    --seed 1337 --img_idx 90 --T 4 \
    --lora_shot_ids 2 3 4 5 --fullft_shot_ids 2 3 4 5 \
    --batch_size 1 --num_workers 4 --gpu 0 --height 1280 --width 720
EOF

echo "All 5 visualization jobs submitted."
