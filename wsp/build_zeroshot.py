#!/usr/bin/env python3
"""Build the zero-shot X2 policy: frozen G1 core + analytic codec, no LoRA.

The released transfer ONNX (`x2_sonic_frozen_g1core_lora_v2.onnx`) is a fused
graph with cleanly separated pieces:

    encoder.*   frozen G1 encoder (verified bit-identical to the checkpoint)
    decoder.*   G1 dynamics decoder with the LoRA adapter merged in
    codec tables (enc_src/enc_a/enc_b, dec_*, d_g1/s_g1/d_x2/s_x2)

Zero-shot = the same graph with `decoder.*` replaced by the frozen `g1_dyn`
weights from the NVIDIA checkpoint. The codec and encoder stay untouched, so
this isolates exactly what the LoRA adapter contributes.

Usage:
    python wsp/build_zeroshot.py \
        --fused models/x2_sonic_frozen_g1core_lora_v2.onnx \
        --checkpoint /path/to/sonic_release_last.pt \
        --out models/x2_sonic_frozen_g1core_zeroshot.onnx
"""
from __future__ import annotations

import argparse
import collections
import pickle

import numpy as np
import onnx
import torch
from onnx import numpy_helper


class _Dummy:
    """Stand-in for TRL/transformers config classes we don't need to read."""

    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)


def _dummy(name: str):
    return type(name.replace(".", "_"), (_Dummy,), {})


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name in ("OrderedDict", "deque"):
            return getattr(collections, name)
        if module.startswith("torch"):
            return super().find_class(module, name)
        return _dummy(f"{module}.{name}")


class _PickleModule:
    Unpickler = _Unpickler

    @staticmethod
    def load(f, **kw):
        return _Unpickler(f, **kw).load()


def load_policy_state_dict(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", pickle_module=_PickleModule, weights_only=False)
    return ck["policy_state_dict"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="models/x2_sonic_frozen_g1core_lora_v2.onnx")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/x2_sonic_frozen_g1core_zeroshot.onnx")
    args = ap.parse_args()

    sd = load_policy_state_dict(args.checkpoint)
    m = onnx.load(args.fused)
    init = {i.name: i for i in m.graph.initializer}

    # Sanity: the encoder in the fused graph must be the frozen G1 encoder.
    for i in (0, 2, 4, 6, 8):
        for t in ("weight", "bias"):
            a = numpy_helper.to_array(init[f"encoder.{i}.{t}"])
            b = sd[f"actor_module.encoders.g1.module.{i}.{t}"].numpy()
            if not np.array_equal(a, b):
                raise SystemExit(f"encoder mismatch at {i}.{t}: max {np.abs(a - b).max()}")

    # Replace only the decoder. Codec + encoder stay bit-identical.
    kept = [i for i in m.graph.initializer if not i.name.startswith("decoder.")]
    for i in (0, 2, 4, 6, 8, 10, 12):
        for t in ("weight", "bias"):
            w = sd[f"actor_module.decoders.g1_dyn.module.{i}.{t}"].numpy()
            kept.append(numpy_helper.from_array(w, name=f"decoder.{i}.{t}"))
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)

    onnx.save(m, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
