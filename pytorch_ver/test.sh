lr="1e-4"

echo "${BASH_VERSION}"
for i in {1..4}
do
    python test_main.py --learning_rate $lr
    echo $i$lr
done
