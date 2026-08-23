uv run python cvae.py --out-dir runs --losses bce --num-workers 4 --anomaly-loss bce

python interpolation_compare.py --latent-dim 32 --pairs 0-1 3-8 4-9 7-2 --steps 11
