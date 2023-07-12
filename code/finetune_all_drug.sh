#!/usr/bin/env bash
#SBATCH --job-name all_drug                # 任务名
#SBATCH --array 0-2                        # 提交 100 个子任务，序号分别为 0,1,2,...99
#SBATCH --gres gpu:a100:1                  # 每个子任务都用x张 A100 GPU
#SBATCH --time 10-16:00:00                   # 子任务 x 天 x 小时就能跑完
#SBATCH --output ../record/%A_%a_out.txt                  # 序输出重定向到

python -u 0_TT_ft_hyper_main.py --souce_num ${SLURM_ARRAY_TASK_ID} \  # pretrain选择哪几个癌种
            --select_drug_method all \                                # 挑选GDSC中所有药物的数据，所以慢
            --store_dir all_drug_finetune #匹配select_drug_method，存那里
