mkdir logs/test_before100000
python main.py --log_folder test_before100000 --max_iter 99999

mkdir logs/test_after100000	
python main.py --log_folder test_after100000 --load_model logs/test_before100000/99999.pt --max_iter 99999

for i in {2..49}
do
	mkdir logs/test_after${i}00000	
    python main.py --log_folder test_after${i}00000 --load_model logs/test_after$((${i}-1))00000/99999.pt --max_iter 99999
done
