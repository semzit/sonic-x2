# M0 — baseline scoreboard (X2, MuJoCo, CPU)

All runs headless, full clip, `--tuning="" --action-clip 20 --freeze-wrist`
except the incumbent (its `bigrun` deploy preset). Metrics are the player's
per-episode summary: joint MAE in rad, max joint error, pelvis-z MAE in m.

| clip | incumbent (native X2) | frozen G1 + LoRA | frozen G1 + codec (zero-shot) |
|---|---|---|---|
| relaxed_walk (14.8 s) | 14.84 s, MAE 0.1771, max 0.957, pz 0.012 | 14.84 s, MAE 0.1181, max 0.706, pz 0.009 | **7.26 s fall**, MAE 0.1323, max 1.566, pz 0.022 |
| idle_stand (9.6 s) | 9.62 s, MAE 0.1979, max 1.327, pz 0.006 | 9.62 s, MAE 0.1556, max 0.787, pz 0.003 | 9.62 s, MAE 0.1923, max 0.788, pz 0.012 |
| gangam_dance (9.2 s) | 9.22 s, MAE 0.2153, max 2.069, pz 0.034 | 9.22 s, MAE 0.1874, max 1.290, pz 0.029 | 9.22 s, MAE 0.2080, max 1.289, pz 0.041 |

## Reading

- **LoRA beats the incumbent on all three clips** — consistent with the paper's
  claim (their OOD reversal; these clips are in-distribution).
- **Zero-shot survives 2 of 3** (idle, dance) and **falls on the walk at 7.26 s**,
  with `pelvis_z` dropping below 0.40 m. That is the paper's "pelvis sink under
  deep flexion / single-support" failure, reproduced locally.
- Zero-shot's MAE on idle/dance is *lower* than the incumbent's even though it
  falls on the walk: geometry survives the codec; the dynamic gap is what breaks.
- This gives E1 a single, concrete target: **keep the walk alive past 14.8 s and
  pull MAE toward LoRA's 0.118**, by editing weights only.

## Reproduce

```bash
./install.sh
# build the zero-shot graph (needs the NVIDIA checkpoint)
python wsp/build_zeroshot.py \
  --checkpoint /path/to/sonic_release_last.pt \
  --out models/x2_sonic_frozen_g1core_zeroshot.onnx

# incumbent
python scripts/eval_x2_mujoco_onnx.py --onnx models/x2_sonic_14000_g1.onnx \
  --motion motions/x2_relaxed_walk.pkl --no-viewer --total-sim-seconds 30

# LoRA
python scripts/eval_x2_mujoco_onnx.py --onnx models/x2_sonic_frozen_g1core_lora_v2.onnx \
  --motion motions/x2_relaxed_walk.pkl --no-viewer --total-sim-seconds 30 \
  --tuning="" --action-clip 20 --freeze-wrist

# zero-shot
python scripts/eval_x2_mujoco_onnx.py --onnx models/x2_sonic_frozen_g1core_zeroshot.onnx \
  --motion motions/x2_relaxed_walk.pkl --no-viewer --total-sim-seconds 30 \
  --tuning="" --action-clip 20 --freeze-wrist
```
