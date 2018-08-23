mkdir logs/test_1e-5_before100000
python test_main.py --learning_rate 1e-5 --log_folder test_1e-5_before100000 --max_iter 99999

mkdir logs/test_1e-5_after100000	
python test_main.py --learning_rate 1e-5 --log_folder test_1e-5_after100000 --load_model logs/test_1e-5_before100000/99999.pt --max_iter 99999

for i in {2..49}
do
	mkdir logs/test_1e-5__after${i}00000	
    python test_main.py --learning_rate 1e-5 --log_folder test_1e-5_after${i}00000 --load_model logs/test_1e-5_after$((${i}-1))00000/99999.pt --max_iter 99999
done
