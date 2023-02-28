cd ..
jac-run scripts/graph/learn_graph_tasks.py \
    --task grail-WN18RR_v1 \
    --epochs 300 \
    --test-interval 5 \
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
    --batch-size 256 \
    --sparsity-loss-ratio 0 \
    --subgraph single \
    --single-link-pred-bridge path \
    --subgraph-size 10 \
    --link-pred-k 3 \
    --resample 100000 \
    --test-subgraph-size 10 \
    --test-batch-size 256 \
    --epoch-size 500 \
    --test-epoch-size 50 \
    --lr 0.001 \
    --aug-node-feature-dim 0 \
    --num-rels 9 \
    --task-is-directed \
    --nlm-reducer mean \
    --dump-dir log/nlm_single/WN18RR_v1