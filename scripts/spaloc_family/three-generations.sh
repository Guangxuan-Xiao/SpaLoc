cd ..
jac-run scripts/graph/learn_graph_tasks.py \
    --task three-generations \
    --epochs 200 \
    --test-interval 20 \
    --early-stop 1e-8 \
    --use-gpu \
    --train-number 20 \
    --test-number-begin 100 \
    --test-number-step 80 \
    --test-number-end 100 \
    --model sparse_nlm \
    --nlm-breadth 3 \
    --nlm-norm sigmoid \
    --train-number $1 \
    --subgraph neighbor \
    --subgraph-size 20 \
    --neighbor-sizes 2_8_16 \
    --resample $1 \
    --sparsity-loss-ratio 0.01 \
    --dump-dir log/sparse_nlm_neighbor/three-generations_N=$1

