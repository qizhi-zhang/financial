# echo "start tensorboard..."
# tensorboard --logdir=./tfboard  --port=6008 &
echo “=========================================================”
echo "使用年度数据:"
sleep 3
python3 run_dataset2.py
clear
sleep 3
echo "=============================================================="
sleep 3
echo "使用季度数据:"
sleep 3
python3 run_dataset3.py
