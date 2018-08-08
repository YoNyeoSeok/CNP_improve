#mkdir logs/1e-5
#python main.py --learning_rate 1e-5 --log_folder 1e-5 --max_iter 99999 

#mkdir logs/1e-5_after100000	
#python main.py --learning_rate 1e-5 --log_folder 1e-5_after100000 --load_model logs/1e-5/99999.pt --max_iter 99999

#for i in 2 3 4 5 6 7 8 9 10
#do
#	mkdir logs/1e-5_after${i}00000	
#	python main.py --learning_rate 1e-5 --log_folder 1e-5_after${i}00000 --load_model logs/1e-5_after$((${i}-1))00000/99999.pt --max_iter 99999
#done

#for i in 11 12 13 14 15 16 17 18 19
#do
#	mkdir logs/1e-5_after${i}00000	
#    python main.py --learning_rate 1e-5 --log_folder 1e-5_after${i}00000 --load_model logs/1e-5_after$((${i}-1))00000/99999.pt --max_iter 99999
#done

for i in 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
do
	mkdir logs/1e-5_after${i}00000	
    python main.py --learning_rate 1e-5 --log_folder 1e-5_after${i}00000 --load_model logs/1e-5_after$((${i}-1))00000/99999.pt --max_iter 99999
done
