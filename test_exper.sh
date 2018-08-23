lr="1e-4"

mkdir logs/test_${lr}_before100000
python test_main.py --learning_rate ${lr} --log_folder test_${lr}_before100000 --max_iter 99999

mkdir logs/test_${lr}_after100000	
python test_main.py --learning_rate ${lr} --log_folder test_${lr}_after100000 --load_model logs/test_${lr}_before100000/99999.pt --max_iter 99999

for i in {2..49}
do
	mkdir logs/test_${lr}_after${i}00000	
    python test_main.py --learning_rate ${lr} --log_folder test_${lr}_after${i}00000 --load_model logs/test_${lr}_after$((${i}-1))00000/99999.pt --max_iter 99999
done
