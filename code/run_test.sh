#!/usr/bin/env bash
#SBATCH --job-name model_save                # 任务名
#SBATCH --array 1-21                       # 提交 100 个子任务，序号分别为 0,1,2,...99
#SBATCH --gres gpu:a100:1                  # 每个子任务都用x张 A100 GPU
#SBATCH --time 6-16:00:00                   # 子任务 x 天 x 小时就能跑完
#SBATCH --output ../record/%A_%a_out.txt                  # 序输出重定向到

# aim=test_tcga_fenlei_for_com
# mkdir ../record/${aim}
# gpu_num=5
# for i in {1..20}; #6
# do 
#     a=`expr $i - 1`
#     #echo $i;echo $a;
#     b=`expr $a / ${gpu_num}`
#     CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main.py --pretrain_num 0 \
#         --select_gene_method Percent_sd \
#         --zero_shot_num $i  \
#         --CCL_dataset gdsc1_raw \
#         --select_drug_method overlap \
#         --store_dir test_fenlei_for_com \
#         > ../record/${aim}/$i.txt &
# done

# #分各个癌种pretrain
# aim=test_TT_fenlei_for_com
# mkdir ../record/${aim}
# gpu_num=5
# for i in {2..20};
# do 
#     a=`expr $i - 1`
#     #echo $i;echo $a;
#     b=`expr $a / ${gpu_num}`
#     CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main.py --pretrain_num $i  \
#         --select_gene_method Percent_sd \
#         --zero_shot_num 2  \
#         --CCL_dataset gdsc1_raw \
#         --select_drug_method overlap \
#         --store_dir test_fenlei_for_com \
#         > ../record/${aim}/$i.txt &
# done

function next_task(){
    # echo "now '$1'"
    method_num=$1
    # echo $method_num
    aim=0_tcga_all
    # mkdir ../record/${aim}
    gpu_num=5
    for i in {1..20}; 
    do 
        a=`expr $i - 1`
        #echo $i;echo $a;
        b=`expr $a / ${gpu_num}`
        CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main.py --pretrain_num 0 \
            --tcga_construction raw \
            --CCL_type all_CCL \
            --CCL_construction raw \
            --select_gene_method Percent_sd \
            --zero_shot_num $i  \
            --CCL_dataset gdsc1_raw \
            --select_drug_method overlap \
            --class_num 0 \
            --method_num ${method_num} \
            --label_type PFS \
            --store_dir test \
            1> ../record/${aim}/${method_num}_${i}.txt 2>&1 & #method_TT
    done

    #分各个癌种pretrain
    aim=0_TT_all
    # mkdir ../record/${aim}
    gpu_num=5
    for i in {2..20};
    do 
        a=`expr $i - 2`
        #echo $i;echo $a;
        b=`expr $a / ${gpu_num}`
        CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main.py --pretrain_num $i  \
            --tcga_construction raw \
            --CCL_type all_CCL \
            --CCL_construction raw \
            --select_gene_method Percent_sd \
            --zero_shot_num 2  \
            --CCL_dataset gdsc1_raw \
            --select_drug_method overlap \
            --class_num 0 \
            --method_num ${method_num} \
            --label_type PFS \
            --store_dir test \
            1> ../record/${aim}/${method_num}_${i}.txt 2>&1 &
    done
}

for method in 0 # {1..9};
do
    while true
    do
        process=$(ps -ef | grep 0_TT_ft_hyper_main.py | grep -v "grep" | awk "{print $2}" | wc -l)
        current_time=$(date  "+%Y%m%d-%H%M%S")
        #nvidia-smi
        if [ $process -lt 20 ]; then
            echo "Processed over last task";
            next_task $method;
            echo "Processing next task: $method     ${current_time}";
            sleep 2s;
            break;
        else
            echo "Last process running now      ${current_time}";
            sleep 1h;
        fi
    done
done

#nohup bash run_test.sh 1>aaa.txt 2>&1 &