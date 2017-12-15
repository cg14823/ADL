#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu --gres=gpu:1 --account=comsm0018 -t 0-01:00 --mem=20GB --pty bash
