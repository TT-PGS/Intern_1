# On the root folder, run

source .test_env/bin/activate

python -m main --mode dqn --config ./datasets/input_10_2_4_1.json --dqn-episodes 10 --dqn-model qnet.pt
