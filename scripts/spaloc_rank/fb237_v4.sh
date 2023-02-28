cd ..
jac-run scripts/graph/learn_rank_tasks.py \
    --task grail-fb237_v4 \
    --num-rels 219 \
    --dump-dir log/nlm_rank/fb237_v4 \
    --ranking-loss \
    --ranking-margin 10 \
    --train-num-neg-per-pos 4 \
    --test-num-neg-per-pos 50 \
    --num-workers 20 \
    --epochs 100 \
    --test-interval 2 \
    --save-interval 5 \
    --early-stop 1e-8 \
    --use-gpu \
    --test-number-begin 100 \
    --test-number-step 0 \
    --test-number-end 100 \
    --model nlm \
    --nlm-norm tanh \
    --nlm-depth 6 \
    --nlm-attributes 64 \
    --seed 666 \
    --batch-size 32 \
    --sparsity-loss-ratio 0 \
    --subgraph single \
    --single-link-pred-bridge path \
    --subgraph-size 15 \
    --link-pred-k 3 \
    --resample 100000 \
    --test-subgraph-size 16 \
    --test-batch-size 16 \
    --epoch-size 2000 \
    --test-epoch-size 100000 \
    --lr 0.001 \
    --aug-node-feature-dim 0 \
    --task-is-directed \
    --nlm-reducer mean