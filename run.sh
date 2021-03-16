# https://github.com/mvasil/fashion-compatibility/blob/master/main.py
CUDA_VISIBLE_DEVICES=1 python -u main_polyvore.py --datadir ./data \
 --polyvore_split nondisjoint --name novse_nondisjoint \
 --learned --epochs 15 --num_concepts 5  --embed_loss 5e-4  --mask_loss 5e-4 \
 --batch-size 96 --num-workers 8  | tee logs/novse_nondisjoint.log;\
CUDA_VISIBLE_DEVICES=1 python -u main_polyvore.py --datadir ./data \
 --polyvore_split disjoint --name novse_disjoint \
 --learned --epochs 15 --num_concepts 5  --embed_loss 5e-4  --mask_loss 5e-4 \
 --batch-size 96 --num-workers 8  | tee logs/novse_disjoint.log
