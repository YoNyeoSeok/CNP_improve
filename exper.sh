mkdir logs/1e-5_before100000
python main.py --learning_rate 1e-5 --log_folder 1e-5_before100000 --max_iter 99999 --model_layers 1

mkdir logs/1e-5_after100000	
python main.py --learning_rate 1e-5 --log_folder 1e-5_after100000 --load_model logs/1e-5_before100000/99999.pt --max_iter 99999 --model_layers 1

for i in {2..99}
do
	mkdir logs/1e-5_after${i}00000	
    python main.py --learning_rate 1e-5 --log_folder 1e-5_after${i}00000 --load_model logs/1e-5_after$((${i}-1))00000/99999.pt --max_iter 99999 --model_layers 1
done
