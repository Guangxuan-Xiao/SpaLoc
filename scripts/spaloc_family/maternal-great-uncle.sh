cd ..
jac-run scripts/graph/learn_graph_tasks.py \
    --task maternal-great-uncle \
    --epochs 400 \
    --test-interval 20 \
    --early-stop 1e-8 \
    --use-gpu \
    --test-number-begin 100 \
    --test-number-step 0 \
    --test-number-end 100 \
    --model sparse_nlm \
    --nlm-norm tanh \
    --nlm-depth 5 \
    --builtin-head \
    --seed 42 \
    --batch-size 8 \
    --sparsity-loss-ratio 1e-3 \
    --train-number $1 \
    --subgraph neighbor \
    --subgraph-size 20 \
    --neighbor-sizes 3_3_3 \
    --resample $1 \
    --dump-dir log/sparse_nlm_neighbor/maternal-great-uncle_N=$1

