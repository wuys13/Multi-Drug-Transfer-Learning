#还没写，但tcga不用次次都跑，可以跑一次把模型存下来，之后再挨个判断哪个参数适合哪个癌种

function next_task(){
    # echo "now '$1'"
    method_num=$1
    # echo $method_num
    CCL_type=$2
    tcga_construction=$3
    CCL_construction=$4

    ccl_match="yes"   #"no" "match_zs"
    store_dir="classification"
    select_drug_method="overlap"
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
        CUDA_VISIBLE_DEVICES=$b nohup python -u 0_TT_ft_hyper_main_1.py --pretrain_num $i  \
            --tcga_construction $tcga_construction \
            --CCL_type $CCL_type \
            --CCL_construction $CCL_construction \
            --zero_shot_num 2  \
            --CCL_dataset gdsc1_raw \
            --select_drug_method ${select_drug_method} \
            --class_num 0 \
            --method_num ${method_num} \
            --label_type PFS \
            --ccl_match $ccl_match \
            --store_dir $store_dir \
            1> ../record/${aim}/${method_num}_${i}.txt 2>&1 &
    done

    echo "finish TT task upload"
    sleep 2m
    echo "now for tcga"

    aim=${store_dir}_tcga_$2_${select_drug_method}
    mkdir ../record/${aim}
    gpu_num=5
    for i in {2..20}; 
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
            --CCL_dataset gdsc1_raw \
            --select_drug_method ${select_drug_method} \
            --class_num 0 \
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
for method in {9..9};  
do
for CCL_type in all_CCL #single_CCL
do
    while true
    do
        process=$(ps -ef | grep 0_TT_ft_hyper_main_1.py | grep -v "grep" | awk "{print $2}" | wc -l)
        current_time=$(date  "+%Y%m%d-%H%M%S")
        #nvidia-smi
        if [ $process -lt 36 ]; then
            echo "Processed over last task";
            next_task $method $CCL_type $tcga_construction $CCL_construction;
            echo "Processing next task: $method $CCL_type $tcga_construction $CCL_construction   ${current_time}";
            sleep 2s;
            break;
        else
            echo "Last process running now      ${current_time}";
            sleep 1h;
        fi
    done
done
done
done
done

echo "Finish all !!!!"

# nohup bash run_pretrain.sh 1>ccc.txt 2>&1 &
#_1：dataset数据找index后处理，调用内存小但跑的慢，gpu占用率较低，
#跑分类任务，改overlap和all即可