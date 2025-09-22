#!/bin/bash

#============================= training model ============================
python -m main --mode dqn --config "datasets/9" --dqn-train --dqn-episodes 5 --dqn-model qnet_9.pt
python -m main --mode dqn --config "datasets/18" --dqn-train --dqn-episodes 5 --dqn-model qnet_18.pt
python -m main --mode dqn --config "datasets/27" --dqn-train --dqn-episodes 5 --dqn-model qnet_27.pt
python -m main --mode dqn --config "datasets/36" --dqn-train --dqn-episodes 5 --dqn-model qnet_36.pt
python -m main --mode dqn --config "datasets/45" --dqn-train --dqn-episodes 5 --dqn-model qnet_45.pt
python -m main --mode dqn --config "datasets/54" --dqn-train --dqn-episodes 5 --dqn-model qnet_54.pt
python -m main --mode dqn --config "datasets/63" --dqn-train --dqn-episodes 5 --dqn-model qnet_63.pt
python -m main --mode dqn --config "datasets/72" --dqn-train --dqn-episodes 5 --dqn-model qnet_72.pt
python -m main --mode dqn --config "datasets/81" --dqn-train --dqn-episodes 5 --dqn-model qnet_81.pt
python -m main --mode dqn --config "datasets/90" --dqn-train --dqn-episodes 5 --dqn-model qnet_90.pt
#============================= applied trained models ====================
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_9.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_18.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_27.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_36.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_45.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_54.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_63.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_72.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_81.pt
python -m main --mode dqn --config "datasets/test_90/" --dqn-model qnet_90.pt
#============================= collect data ==============================
python ./metrics/dqn_evaluator_tableII.py
