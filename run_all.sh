#!/bin/bash

python -m main --mode ga --config ./configs/input_10_2_4_1.json --split-mode assign --ga-pop 40 --ga-gen 200 --ga-cx 0.9 --ga-mut 0.2 --ga-tk 5 --ga-seed 42 --ga-verbose --out ./results_ga.json

#### 3. For running SA

python -m main --mode sa --config ./configs/input_10_2_4_1.json --split-mode assign --sa-Tmax 500 --sa-Tthreshold 1 --sa-alpha 0.99 --sa-moves-per-T 50 --sa-seed 42 --out ./results_sa.json

#### 4. For running FCFS

python -m main --mode fcfs --config ./configs/input_10_2_4_1.json --split-mode assign --out ./results_fcfs.json
