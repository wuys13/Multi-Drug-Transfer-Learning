# nohup bash run_pretrain_for_pdr.sh 1>pdr_pretrain.txt 2>&1 &

function next_task(){
    select_drug_method="all"
    # echo "now '$1'"
    method_num=$1
    # echo $method_num
    store_dir="pdr" 
    mkdir ../results/${store_dir}

    aim=${store_dir}_TPP_${select_drug_method}
    mkdir ../record/${aim}
    gpu_param=3 #can be set depending on the number of GPUs, here we use 4 GPUs = 12 / 3
    for i in {0..13}; 
    do 
        a=`expr $i - 1`
        b=`expr $a / ${gpu_param}`
        CUDA_VISIBLE_DEVICES=$b nohup python -u P_MDL.py --pretrain_num $i  \
            --zero_shot_num 2  \
            --CCL_dataset gdsc1_rebalance \
            --select_drug_method $select_drug_method \
            --method_num ${method_num} \
            --store_dir $store_dir \
            1> ../record/${aim}/${method_num}_${i}.txt 2>&1 &
    done
}


for method in {0..7}; 
do
    while true
    do
        process=$(ps -ef | grep P_MDL | grep -v "grep" | awk "{print $2}" | wc -l)
        current_time=$(date  "+%Y%m%d-%H%M%S")
        #nvidia-smi
        if [ $process -lt 20 ]; then
            echo "Process < 21, finished last task!"
            next_task $method ;
            echo "Processing next task: $method   ${current_time}";
            break;
        else
            echo "Running task num: ${process}      ${current_time}";
            sleep 12h;
        fi
    done
done

echo "Finish all !!!!"
