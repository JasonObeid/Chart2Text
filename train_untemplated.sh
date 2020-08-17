#!/bin/bash
#SBATCH --account=def-enamul
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=0-12:00

module load python/3.7 arch/avx512 StdEnv/2018.3

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch--1.5.0 --no-index
pip install torchvision --no-index
pip install numpy --no-index
pip install spacy --no-index
pip install ../en_core_web_md-2.2.5.tar.gz
python model/train.py \
    --model_path "experiments" \
    --exp_name "chart2text" \
    --exp_id "run1" \
    --train_cs_table_path data_untemplated/train/trainData.txt.pth \
    --train_sm_table_path data_untemplated/train/trainData.txt.pth \
    --train_sm_summary_path data_untemplated/train/trainSummary.txt.pth \
    --valid_table_path data_untemplated/valid/validData.txt.pth \
    --valid_summary_path data_untemplated/valid/validSummary.txt.pth \
    --cs_step True \
    --lambda_cs "1" \
    --sm_step True \
    --lambda_sm "1" \
    --label_smoothing 0.05 \
    --sm_step_with_cc_loss False \
    --sm_step_with_cs_proba False \
    --share_inout_emb True \
    --share_srctgt_emb False \
    --emb_dim 512 \
    --enc_n_layers 1 \
    --dec_n_layers 6 \
    --dropout 0.1 \
    --save_periodic 40 \
    --batch_size 6 \
    --beam_size 4 \
    --epoch_size 1000 \
    --max_epoch 81 \
    --eval_bleu True \
    --sinusoidal_embeddings True \
    --encoder_positional_emb True \
    --gelu_activation True \
    --validation_metrics valid_mt_bleu