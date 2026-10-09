# SURF

```bash
pip install mbrl==0.1.5 --no-deps && pip install -r requirements.txt
```

Run from the repo root. Results go to `results/`.

**FrozenLake** (`--change-p` = post-change p′: 0.9 / 0.7 / 0.5 / 0.3 / 0.1)
```bash
python FrozenLake/sweep_unfrozen_layers.py --n-unfrozen 1 --trials 99 \
    --cvar-alpha 0 --clip-delta-n 2 --change-p 0.7 --out-dir results/fl_clip2_n99
python FrozenLake/summarize_clip2_n99.py
```

**NS-Bridge**
```bash
(cd Bridge && bash run_bridge_ablation.sh)
python Bridge/summarize_bridge_ablation.py
```

**Pendulum**
```bash
(cd pendulum-reproduce && python run_main.py)
```

**Reacher**: see `Reacher/README.md`.
