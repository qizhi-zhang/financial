echo "使用年度数据:"
sleep 3
python3 run_dataset2.py --local 0 --job_name worker --task_index 0 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 1 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 2 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 3 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 4 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 5 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 6 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 7 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 8 &
python3 run_dataset2.py --local 0 --job_name worker --task_index 9 &
python3 run_dataset2.py --local 0 --job_name ps --task_index 0 
sleep 6
ps -ef | grep "python3" | grep -v grep | awk '{print $2}' | xargs -r  kill -9 
clear
echo "=============================================================="
echo "使用季度数据:"
sleep 3
python3 run_dataset3.py --local 0 --job_name worker --task_index 0 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 1 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 2 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 3 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 4 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 5 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 6 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 7 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 8 &
python3 run_dataset3.py --local 0 --job_name worker --task_index 9 &
python3 run_dataset3.py --local 0 --job_name ps --task_index 0 
ps -ef | grep "python3" | grep -v grep | awk '{print $2}' | xargs -r  kill -9
