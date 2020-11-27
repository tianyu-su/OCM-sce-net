# https://github.com/mvasil/fashion-compatibility/blob/master/main.py
python main_poluvore.py --datadir ${POLY} \
 --polyvore_split nondisjoint --name polyvore_nondisjoint \
 --learned --epochs 100 --num_concepts 5  --embed_loss 5e-4  --mask_loss 5e-4 \
 --batch-size 78 --num-workers 8