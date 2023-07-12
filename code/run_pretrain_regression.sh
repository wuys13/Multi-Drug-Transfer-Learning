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
    CCL_type=$2
    tcga_construction=$3
    CCL_construction=$4

    ccl_match="yes"   #"no" "match_zs"
    store_dir="regression_more_params"
    select_drug_method="all"
    # if [ $ccl_match == "yes" ]; then
    #     store_dir="test"
    # else
    #     store_dir="not_match"
    # fi

    
    #分各个癌种pretrain
    aim=${store_dir}_TT_$2_${select_drug_method}
    mkdir ../record/${aim}
    gpu_num=5
    for i in {2..20};
    do 
        a=`expr $i - 1`
        #echo $i;echo $a;
        b=`expr $a / ${gpu_num}`
        # if [ $b -gt 1 ]; then
        #     b=`expr $b - 2`
        # fi
        CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main_1.py --pretrain_num $i  \
            --tcga_construction $tcga_construction \
            --CCL_type $CCL_type \
            --CCL_construction $CCL_construction \
            --zero_shot_num 2  \
            --CCL_dataset gdsc1_rebalance_regression \
            --select_drug_method ${select_drug_method} \
            --class_num 1 \
            --method_num ${method_num} \
            --label_type PFS \
            --ccl_match $ccl_match \
            --store_dir $store_dir \
            1> ../record/${aim}/${method_num}_${i}.txt 2>&1 &
       
        # while true
        # do
        #     process=$(ps -ef | grep 0_TT_ft_hyper_main | grep -v "grep" | awk "{print $2}" | wc -l)
        #     current_time=$(date  "+%Y%m%d-%H%M%S")
        #     #nvidia-smi
        #     if [ $process -lt 11 ]; then
        #         echo $i
        #         echo "Now process num: ${process} < 11     ${current_time}";
        #         break;
        #     else
        #         echo "Now process num: ${process}      ${current_time}";
        #         sleep 24h;
        #     fi
        # done
    done

    echo "finish TT task upload"
    sleep 2m
    echo "now for tcga"
    

    aim=${store_dir}_tcga_$2_${select_drug_method}
    mkdir ../record/${aim}
    gpu_num=5
    for i in {2..2}; 
    do 
        a=`expr $i - 1`
        #echo $i;echo $a;
        b=`expr $a / ${gpu_num}`
        CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main_1.py --pretrain_num 0 \
            --tcga_construction $tcga_construction \
            --CCL_type $CCL_type \
            --CCL_construction $CCL_construction \
            --select_gene_method Percent_sd \
            --zero_shot_num $i  \
            --CCL_dataset gdsc1_rebalance_regression \
            --select_drug_method ${select_drug_method} \
            --class_num 1 \
            --method_num ${method_num} \
            --label_type PFS \
            --ccl_match $ccl_match \
            --store_dir $store_dir \
            1> ../record/${aim}/${method_num}_${i}.txt 2>&1 & #method_TT
    done
}

for CCL_construction in raw #pseudo
do
for tcga_construction in raw #pseudo
do
for method in {9..10};  
do
for CCL_type in all_CCL #single_CCL
do
    while true
    do
        process=$(ps -ef | grep 0_TT_ft_hyper_main | grep -v "grep" | awk "{print $2}" | wc -l)
        current_time=$(date  "+%Y%m%d-%H%M%S")
        #nvidia-smi
        if [ $process -lt 4 ]; then
            echo "Process < 4, finished last task!";
            next_task $method $CCL_type $tcga_construction $CCL_construction;
            echo "Processing next task: $method $CCL_type $tcga_construction $CCL_construction   ${current_time}";
            break;
        else
            echo "Last process running now  ${process}     ${current_time}";
            sleep 12h;
        fi
    done
done
done
done
done

echo "Finish all !!!!"

#nohup bash run_pretrain_regression.sh 1>rrr.txt 2>&1 &
