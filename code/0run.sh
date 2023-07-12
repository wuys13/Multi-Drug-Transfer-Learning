#!/usr/bin/env bash
#SBATCH --job-name model_save                # 任务名
#SBATCH --array 1-21                       # 提交 100 个子任务，序号分别为 0,1,2,...99
#SBATCH --gres gpu:a100:1                  # 每个子任务都用x张 A100 GPU
#SBATCH --time 6-16:00:00                   # 子任务 x 天 x 小时就能跑完
#SBATCH --output ../record/%A_%a_out.txt                  # 序输出重定向到
aim=test
mkdir ../record/${aim}
gpu_num=5
for i in {2..20};
do 
    a=`expr $i - 2`
    #echo $i;echo $a;
    b=`expr $a / ${gpu_num}`
    CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main.py --pretrain_num $i \
        --select_gene_method Percent_sd \
        --zero_shot_num 2  \
        --CCL_dataset gcp_combine \
        --select_drug_method overlap \
        --store_dir test \
        > .../record/${aim}/$i.txt &
done

