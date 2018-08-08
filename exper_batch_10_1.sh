mkdir logs/1e-5_batch_10_1
python main.py --learning_rate 1e-5 --log_folder 1e-5_batch_10_1 --max_iter 99999 

mkdir logs/1e-5_batch_10_1_after100000	
python main.py --learning_rate 1e-5 --log_folder 1e-5_batch_10_1_after100000 --load_model logs/1e-5_batch_10_1/99999.pt --max_iter 99999

for i in 2 3 4 5 6 7 8 9 10
do
	mkdir logs/1e-5_batch_10_1_after${i}00000	
	python main.py --learning_rate 1e-5 --log_folder 1e-5_batch_10_1_after${i}00000 --load_model logs/1e-5_batch_10_1_after$((${i-1}-1))00000/99999.pt --max_iter 99999
done

