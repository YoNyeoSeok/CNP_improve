lr="1e-4"
data_source="gp1d"
max_epoch=1000000
interval=10000
log_folder="${data_source}_${lr}/"

sudo rm -r logs/${data_source}_${lr}
mkdir logs/${data_source}_${lr}
python main.py \
    --datasource ${data_source} --not_disjoint_data --random_sample --num_samples_range 1 51 --input_range -2 2 \
    --learning_rate ${lr} \
    --batch_size 32 --interval ${interval} --max_epoch ${max_epoch} --log_folder ${log_folder} \
    --window_range -2 2 --window_step_size .1 --fig_show --log --gpu 1 \
    
#
#mkdir logs/test_${lr}_after100000	
#python test_main.py --learning_rate ${lr} --log_folder test_${lr}_after100000 --load_model logs/test_${lr}_before100000/99999.pt --max_iter 99999
#
#for i in {2..49}
#for i in {50..99}
#do
#	mkdir logs/test_${lr}_after${i}00000	
#    python test_main.py --learning_rate ${lr} --log_folder test_${lr}_after${i}00000 --load_model logs/test_${lr}_after$((${i}-1))00000/99999.pt --max_iter 99999
#done
