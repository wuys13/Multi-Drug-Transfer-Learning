#nohup bash run_pdr_task.sh 1>pdr_task.txt 2>&1 &

function next_task(){
    select_drug_method="all"
    # echo "now '$1'"
    method_num=$1
    # echo $method_num
    store_dir="pdr_task" 
    mkdir ../results/${store_dir}
    
    aim=${store_dir}
    mkdir ../records/${aim}
    gpu_num=5
    for i in {2..20};
    do 
        if [ $i -gt 5 ]; then
            echo  "Can choose to wait for 5 min ~~~"
            sleep 5m
            echo  "Sleep over. Now for $i"
        fi
        a=`expr $i - 1`
        #echo $i;echo $a;
        b=`expr $a / ${gpu_num}`
        #b=3
        CUDA_VISIBLE_DEVICES=$b nohup python -u PDR_task.py --pretrain_num $i  \
            --zero_shot_num 2  \
            --CCL_dataset gdsc1_rebalance \
            --select_drug_method $select_drug_method \
            --method_num ${method_num} \
            --store_dir $store_dir \
            1> ../records/${aim}/${method_num}_${i}.txt 2>&1 &
    done
}


for method in {7..7}; 
do
    while true
    do
        process=$(ps -ef | grep PDR_task | grep -v "grep" | awk "{print $2}" | wc -l)
        current_time=$(date  "+%Y%m%d-%H%M%S")
        #nvidia-smi
        if [ $process -lt 1 ]; then
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

