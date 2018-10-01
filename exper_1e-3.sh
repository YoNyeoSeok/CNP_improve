mkdir logs/1e-3
python main.py --learning_rate 1e-3 --log_folder 1e-3 --max_iter 99999 

mkdir logs/1e-3_after100000	
python main.py --learning_rate 1e-3 --log_folder 1e-3_after100000 --load_model logs/1e-3/99999.pt --max_iter 99999

for i in {2..99}
do
	mkdir logs/1e-3_after${i}00000	
	python main.py --learning_rate 1e-3 --log_folder 1e-3_after${i}00000 --load_model logs/1e-3_after$((${i}-1))00000/99999.pt --max_iter 99999
done
