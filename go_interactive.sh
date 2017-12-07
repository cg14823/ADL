#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu --gres=gpu:1 -t 0-02:00 --mem=100GB --pty bash