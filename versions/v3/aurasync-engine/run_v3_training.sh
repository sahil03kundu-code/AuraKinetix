#!/bin/bash
# Resumable: re-run after an interruption; finished jobs are skipped, unfinished ones resume.
cd "$(dirname "$0")"
mkdir -p checkpoints/v3
fin() { grep -q "^done" checkpoints/v3/$1.log 2>/dev/null; }
ev() { fin $1 || python3 train_v3.py --steps 2000 --seed $2 --name $1 $3 >> checkpoints/v3/$1.log 2>&1; }
fr() { fin $1 || python3 train_frame.py --steps 5000 --batch 32 --seed $2 --threads 1 --out checkpoints/v3/$1.pt >> checkpoints/v3/$1.log 2>&1; }
( ev event_s2 2; ev event_s0 0; fr frame_s1 1 ) &
( ev event_s1 1; ev eventonly_s0 0 --no_aps; fr frame_s2 2 ) &
wait
