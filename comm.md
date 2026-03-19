# 1. Générer dataset amélioré
python simulated_data_scalable.py --num-nodes 100 --num-samples 10000 \
    --v2 --seasonal-noise --correlated-features --multi-root-prob 0.15

# 2. Entraîner avec Memory Module
python train_sta.py --dataset scalable_dataset_N100_v2.pt \
    --use-memory --use-severity-weights --curriculum \
    --out-checkpoint sta_sota_full.pt

# 3. Évaluer avec breakdown
python evaluate.py --checkpoint checkpoints/sta_sota_full.pt \
    --dataset scalable_dataset_N100_v2.pt --breakdown