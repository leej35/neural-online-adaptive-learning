#!/usr/bin/env bash
# source activate py37torch13v3

cd ../

data_path="/datapath/mimic.sequence/"
python_path="/afs/cs.pitt.edu/usr0/jel158/anaconda3/envs/py37torch13v3/bin/python"
base_path="/afs/cs.pitt.edu/usr0/jel158/public/project/ts_dataset/"
source activate py37torch13v3

curriculum="False"
curriculum_rate="1.005"
lr_scheduler="False"
lr_scheduler_numiter="15"
lr_scheduler_multistep="False"
lr_scheduler_epochs="15 45 75 105 135 180 225"
lr_scheduler_mult="0.5"

mkdir -p "logs"
mkdir -p "trained"

window_hr="$1"
model_type="$2"
model_name=${model_type}
split_id="$3"

num_folds="5"
max_epoch="5000"
valid_every="5"
patient_stop="5"
filename="${model_name}_w${window_hr}_s${split_id}"
curtime=$(date +"%Y_%m_%d_%H_%M_%S")
outdir="trained/fml_${curtime}_${model_name}_w${window_hr}_s${split_id}"

source scripts/model_opts.sh

mkdir -p ${outdir}
CUDA_VISIBLE_DEVICES=0 "${python_path}" main.py \
    --epoch "${max_epoch}" \
    --validate-every "${valid_every}" \
    --save-every -1 \
    --model-prefix "${outdir}/${filename}" \
    --num-workers 0 \
    --print-every 1 \
    --n-layers 1 \
    --patient-stop "${patient_stop}" \
    --data-name "mimic3" \
    --base-path "${data_path}"\
    --window-hr-x "${window_hr}" \
    --window-hr-y "${window_hr}" \
    ${model_opt} \
    --model-name "${model_name}" \
    --split-id "${split_id}" \
    --num-folds "${num_folds}" \
    --remapped-data \
    --use-mimicid \
    --cuda \
    --labrange \
    --pred-normal-labchart \
    --opt-str "_minsd_2_maxsd_20_sv" \
    --force-auroc \
    --prior-from-mimic \
    --code-name "m2x20_predall_v5_hyper" \
    --hidden-dim 128 \
    --embedding-dim 64 \
    --learning-rate 0.005 \
    --batch-size 128 \
    --elapsed-time \
    --force-epoch \
    --target-auprc \
    $4 \
    | tee "logs/log_${filename}.txt" "${outdir}/log_${filename}.txt" --ignore-interrupts

cd -