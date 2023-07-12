#!/usr/bin/env bash
#SBATCH --job-name model_save                # 任务名
#SBATCH --array 1-21                       # 提交 100 个子任务，序号分别为 0,1,2,...99
#SBATCH --gres gpu:a100:1                  # 每个子任务都用x张 A100 GPU
#SBATCH --time 6-16:00:00                   # 子任务 x 天 x 小时就能跑完
#SBATCH --output ../record/%A_%a_out.txt                  # 序输出重定向到

python -u 0_TT_ft_hyper_main.py --pretrain_num ${SLURM_ARRAY_TASK_ID} \  # pretrain选择哪几个癌种
            --select_gene_method Percent_sd \
            --zero_shot_num 0  \ # 只有当pretrain_num = 0 时有意义，此时args.pretrain_dataset变为TCGA，需要根据zero_shot_num选定要测试的癌种数据
            --CCL_dataset GDSC1_rec \ # 存到../data/raw_dat/CCL_dataset/
            --select_drug_method overlap \                   # 挑选GDSC和TCGA都有的药物数据，大部分癌种map到的不多
            --store_dir test 
