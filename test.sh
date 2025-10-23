#!/bin/bash

TRAINING_TESTING_DATASET="datasets/90/"

#============================= training model ============================
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 9 --dqn-model qnet_9.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 18 --dqn-model qnet_18.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 27 --dqn-model qnet_27.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 36 --dqn-model qnet_36.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 45 --dqn-model qnet_45.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 54 --dqn-model qnet_54.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 63 --dqn-model qnet_63.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 72 --dqn-model qnet_72.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 81 --dqn-model qnet_81.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-train --dqn-episodes 90 --dqn-model qnet_90.pt
#============================= applied trained models ====================
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_9.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_18.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_27.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_36.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_45.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_54.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_63.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_72.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_81.pt
python -m main --mode dqn --config ${TRAINING_TESTING_DATASET} --dqn-model qnet_90.pt
#============================= collect data ==============================
python ./metrics/dqn_evaluator_tableII.py
python draw_gantt.py

rm envs.log trace_steps.jsonl runner.log