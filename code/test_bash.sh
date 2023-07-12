#!/usr/bin/env bash
# while true
# do
#     process=$(ps -ef | grep 0_TT_ft_hyper_main.py | grep -v "grep" | awk "{print $2}")

#     if [ "$process" == "" ]; then
#         echo "process over";
#         break;
#     else
#         # echo "process run";
#         sleep 1h;
#     fi
# done


# a="a"
# b=2
# function next_task(){
#     c=$1_$2
#     echo $c
# }
# next_task $a $b

# c=${a}_${b}
# echo $c
# echo "${a}_${b}"

# ccl_match="yes"
# if [ $ccl_match == "yes" ]; then
#     store_dir="test"
# else
#     store_dir="not_match"
# fi

# echo $store_dir

for method in {10..0}
do
echo $method
done

# for CCL_type in single_CCL #all_CCL
# do
# echo $CCL_type
# done

# process=$(ps -ef | grep 0_TT_ft_hyper_main.py | grep -v "grep" | awk "{print $2}" | wc -l)
# if [ $process -lt 10 ]; then
#     echo "Processed over last task";
# fi