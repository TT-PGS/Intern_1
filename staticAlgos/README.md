# Using these commands to run, for now, we aim to improve the algos, will enhance the main.py later

source .test_env/bin/activate

python3 -m staticAlgos.FCFS --config ./datasets/input_10_2_4_1.json --mode timespan

python3 -m staticAlgos.SA --config ./datasets/input_10_2_4_1.json --mode timespan

python3 -m staticAlgos.GA --config ./datasets/input_10_2_4_1.json --mode timespan
