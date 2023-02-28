cd ..
jac-run scripts/graph/learn_graph_tasks.py \
    --task grandparents \
    --epochs 300 \
    --test-interval 20 \
    --early-stop 1e-7 \
    --use-gpu \
    --test-number-begin 20 \
    --test-number-step 80 \
    --test-number-end 100 \
    --model sparse_nlm \
    --nlm-norm sigmoid \
    --nlm-depth 5 \
    --sparsity-loss-ratio 0.01 \
    --builtin-head \
    --seed 1 \
    --train-number $1 \
    --subgraph neighbor \
    --subgraph-size 20 \
    --neighbor-sizes 2_8_16 \
    --resample $1 \
    --dump-dir log/sparse_nlm_neighbor/grandparents_N=$1

