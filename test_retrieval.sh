# yga:/home/share/yaoguangan/benpao/ckpts_tmp/
# s9: /home/share/sutianyu/final_ckpts/

#nondisjoint:
echo "nondisjoint"
CUDA_VISIBLE_DEVICES=1 python -u  evaluation_retrieval.py  --name novse_nondisjoint --datadir ./data --learned --l2_embed --learned_metric --test --resume /home/share/sutianyu/final_ckpts/SCE-NET/runs/novse_nondisjoint/model_best.pth.tar | tee  output_nondisjoint.log
#disjoint:
echo "disjoint"
CUDA_VISIBLE_DEVICES=1 python -u  evaluation_retrieval.py  --name novse_disjoint --polyvore_split disjoint --datadir ./data --learned --l2_embed --learned_metric --test --resume /home/share/sutianyu/final_ckpts/SCE-NET/runs/novse_disjoint/model_best.pth.tar | tee output_disjoint.log




CUDA_VISIBLE_DEVICES=1 python -u evaluation_retrieval.py --datadir ./data \
 --polyvore_split nondisjoint --name novse_nondisjoint \
 --learned --epochs 15 --num_concepts 5  --embed_loss 5e-4  --mask_loss 5e-4 \
 --batch-size 96 --num-workers 8  --test --resume /home/share/sutianyu/final_ckpts/SCE-NET/runs/novse_nondisjoint/model_best.pth.tar | tee  output_nondisjoint.log


CUDA_VISIBLE_DEVICES=1 python -u evaluation_retrieval.py --datadir ./data \
 --polyvore_split disjoint --name novse_disjoint \
 --learned --epochs 15 --num_concepts 5  --embed_loss 5e-4  --mask_loss 5e-4 \
 --batch-size 96 --num-workers 8  --test --resume /home/share/sutianyu/final_ckpts/SCE-NET/runs/novse_disjoint/model_best.pth.tar | tee output_disjoint.log

