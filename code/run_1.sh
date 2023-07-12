#!/usr/bin/env bash
#SBATCH --job-name model_save                # 任务名
#SBATCH --array 1-21                       # 提交 100 个子任务，序号分别为 0,1,2,...99${SLURM_ARRAY_TASK_ID}
#SBATCH --gres gpu:a100:1                  # 每个子任务都用x张 A100 GPU
#SBATCH --time 6-16:00:00                   # 子任务 x 天 x 小时就能跑完
#SBATCH --output ../record/%A_%a_out.txt                  # 序输出重定向到


# #generate model0: pdr
# CUDA_VISIBLE_DEVICES=0 nohup python -u 0_TT_all_drug_main.py --pretrain_num 2  \
#             --zero_shot_num 2  \
#             --CCL_dataset gdsc1_raw \
#             --select_drug_method all \
#             --class_num 0 \
#             --method_num 10 \
#             --label_type PFS \
#             --ccl_match yes \
#             --store_dir all_drug \
#             --iteration_method source \
#             1> bbb.txt 2>&1 &

# #1test pdr task:use generative model0
# python -u 0_pdr_task.py --pretrain_num 2  \
#             --zero_shot_num 2  \
#             --CCL_dataset gdsc1_raw \
#             --select_drug_method  all \
#             --class_num 0 \
#             --method_num 10 \
#             --label_type PFS \
#             --store_dir all_drug

# for a in gdsc1_raw gdsc1_rebalance; 
# do
# for b in overlap all; 
# do
# CUDA_VISIBLE_DEVICES=1  nohup python -u 0_TT_ft_hyper_main_1.py --pretrain_num 0 \
#             --tcga_construction raw \
#             --CCL_type all_CCL \
#             --CCL_construction raw \
#             --select_gene_method Percent_sd \
#             --zero_shot_num 3  \
#             --CCL_dataset $a \
#             --select_drug_method $b \
#             --class_num 0 \
#             --method_num 10 \
#             --label_type PFS \
#             --ccl_match yes \
#             --class_num 1 \
#             --store_dir regression \
#             1> ../record/03_${a}_${b}.txt 2>&1 & 
# done
# done

# for a in gdsc1_raw gdsc1_rebalance; 
# do
# for b in overlap all; 
# do
# CUDA_VISIBLE_DEVICES=1  nohup python -u 0_TT_ft_hyper_main_1.py --pretrain_num 0 \
#             --tcga_construction raw \
#             --CCL_type all_CCL \
#             --CCL_construction raw \
#             --select_gene_method Percent_sd \
#             --zero_shot_num 20  \
#             --CCL_dataset $a \
#             --select_drug_method $b \
#             --class_num 0 \
#             --method_num 10 \
#             --label_type PFS \
#             --ccl_match yes \
#             --class_num 1 \
#             --store_dir regression \
#             1> ../record/020_${a}_${b}.txt 2>&1 & 
# done
# done

# #test one method
# CUDA_VISIBLE_DEVICES=0  python -u 0_TT_ft_hyper_main.py --pretrain_num 2 \
#             --tcga_construction raw \
#             --CCL_type all_CCL \
#             --CCL_construction raw \
#             --select_gene_method Percent_sd \
#             --zero_shot_num 0  \
#             --CCL_dataset gdsc1_raw \
#             --select_drug_method all \
#             --class_num 0 \
#             --method_num 10 \
#             --label_type PFS \
#             --ccl_match yes \
#             --store_dir all_test \
#             --iteration_method source

# CUDA_VISIBLE_DEVICES=3 nohup python -u 0_TT_ft_hyper_main_1.py --pretrain_num 0 \
#             --tcga_construction raw \
#             --CCL_type all_CCL \
#             --CCL_construction raw \
#             --select_gene_method Percent_sd \
#             --zero_shot_num 17  \
#             --CCL_dataset gdsc1_raw \
#             --select_drug_method overlap \
#             --class_num 0 \
#             --method_num 2 \
#             --label_type PFS \
#             --ccl_match yes \
#             --store_dir classification \
#             1> ../record/classification_tcga_all_CCL_overlap/2_17.txt 2>&1 & 

CUDA_VISIBLE_DEVICES=0  python -u 0_TT_ft_hyper_main_2.py --pretrain_num 2 \
            --tcga_construction raw \
            --CCL_type all_CCL \
            --CCL_construction raw \
            --select_gene_method Percent_sd \
            --zero_shot_num 0  \
            --CCL_dataset gdsc1_raw \
            --select_drug_method overlap \
            --class_num 0 \
            --method_num 9 \
            --label_type PFS \
            --ccl_match yes \
            --store_dir overlap_TT
            
#nohup bash run_1.sh 1>bbb.txt 2>&1 &