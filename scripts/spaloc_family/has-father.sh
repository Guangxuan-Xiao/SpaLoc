cd ..
jac-run scripts/graph/learn_graph_tasks.py \
    --task has-father \
    --epochs 100 \
    --test-interval 5 \
    --use-gpu \
    --test-number-begin 100 \
    --test-number-step 0 \
    --test-number-end 100 \
    --model sparse_nlm \
    --nlm-norm tanh \
    --nlm-depth 5 \
    --sparsity-loss-ratio 0.01 \
    --builtin-head \
    --seed 1 \
    --train-number $1 \
    --subgraph neighbor \
    --subgraph-size 20 \
    --neighbor-sizes 3_3_3 \
    --resample $1 \
    --dump-dir log/sparse_nlm_neighbor/has-father_N=$1

