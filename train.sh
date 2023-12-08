# python3 main.py --batch_size 18 --accelerator gpu --devices 1 --max_epochs 5 -lr 1e-4 --val_check_interval 0.25
export PATH="/mnt/AM4_disk3/chenwuchen/env/py_gcc82/py_38/envs/punc/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/AM4_disk3/chenwuchen/env/py_gcc82/py_38/envs/punc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/mnt/AM4_disk3/chenwuchen/env/py_gcc82/py_38/ext_lib:$LD_LIBRARY_PATH"
source /mnt/AM4_disk3/chenwuchen/env/py_gcc82/py_38/envs/punc/bin/activate


log_dir=./log
log=${log_dir}/log_$(date +%y%m%d_%H%M%S)

# mdl=mdl/zhpr_test_mdl_230906
mdl=mdl/chinese_mobilebert_base_f2

python3 -m src.bin.train --base_mdl $mdl --batch_size 36 --max_epochs 5 -lr 1e-4 >& ${log}
