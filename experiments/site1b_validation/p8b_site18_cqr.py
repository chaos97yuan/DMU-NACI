"""Site 18 -- canonical CQR under the rigorous_v2 protocol (for Table Y parity).

Trains the frozen MultiScaleTCN trunk + 3-quantile pinball head on the SAME
rigorous_v2 gradient rows (targets < 2024-07-17), internal-tail early stopping,
conformalizes on the rigorous_v2 CALIB span and evaluates on the untouched 2025
CROSS span. 5 matched seeds. Report-only.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]  # release: repo-relative (was absolute path)
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("v2", ROOT / "experiments" / "rigorous_v2" / "common_v2.py")
V2 = importlib.util.module_from_spec(spec)
sys.modules["v2x"] = V2
spec.loader.exec_module(V2)

from src.models.multiscale_tcn import MultiScaleTCN  # noqa: E402

OUT = ROOT / "results" / "rigorous_v2" / "s7_final_eval" / "site18_cqr"
S2 = ROOT / "results" / "rigorous_v2" / "s2_final_mean"
S6 = ROOT / "results" / "rigorous_v2" / "s6_gru_select"
P18 = 5.2


class CQRNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = MultiScaleTCN(dropout=0.1)
        self.head = torch.nn.Linear(64, 3)

    def forward(self, x):
        return self.head(self.trunk.forward_features(x))


def train_cqr(grad, es, sc, seed):
    torch.manual_seed(seed % (2 ** 31))
    model = CQRNet().to(V2.DEVICE)
    x = torch.as_tensor(V2.apply_scaler(grad["x_raw"], sc), dtype=torch.float32, device=V2.DEVICE)
    y = torch.as_tensor(((grad["y_kw"] - sc["target_mean"]) / sc["target_std"]).astype(np.float32),
                        device=V2.DEVICE)
    xv = torch.as_tensor(V2.apply_scaler(es["x_raw"], sc), dtype=torch.float32, device=V2.DEVICE)
    yv = np.asarray(es["y_kw"], float)
    qs = torch.tensor([0.05, 0.5, 0.95], device=V2.DEVICE).view(1, -1)
    rng = np.random.default_rng(58003)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    @torch.no_grad()
    def es_rmse():
        model.eval()
        p = model(xv).detach().cpu().numpy().astype(float)
        model.train()
        return float(np.sqrt(np.mean((p[:, 1] * sc["target_std"] + sc["target_mean"] - yv) ** 2)))

    best, best_state, no = float("inf"), None, 0
    for epoch in range(100):
        perm = rng.permutation(len(y))
        for s0 in range(0, len(y), 512):
            idx = perm[s0:s0 + 512]
            opt.zero_grad()
            err = y[idx].unsqueeze(1) - model(x[idx])
            loss = torch.mean(torch.mean(torch.maximum(qs * err, (qs - 1) * err), dim=1))
            loss.backward()
            opt.step()
        vr = es_rmse()
        if vr < best - 1e-6:
            best, no = vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no += 1
            if no >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def quantiles(model, x_raw, sc):
    xs = V2.apply_scaler(x_raw, sc)
    outs = []
    for s0 in range(0, len(xs), 4096):
        xb = torch.as_tensor(xs[s0:s0 + 4096], dtype=torch.float32, device=V2.DEVICE)
        outs.append(model(xb).detach().cpu().numpy().astype(float))
    return np.concatenate(outs) * sc["target_std"] + sc["target_mean"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sc_raw = json.loads((S2 / "scaler.json").read_text())
    sc = {"mean": np.array(sc_raw["mean"], float), "std": np.array(sc_raw["std"], float),
          "target_mean": float(sc_raw["target_mean"]), "target_std": float(sc_raw["target_std"])}
    y2024 = V2.load_year_windows(2024)
    y2025 = V2.load_year_windows(2025)

    def rows_by_target(part, lo, hi):
        m = V2.target_mask(part["targets"], lo, hi)
        return {k: v[m] for k, v in part.items() if k in ("x_raw", "y_kw", "origins", "targets")}

    grad = rows_by_target(y2024, *V2.MEAN_GRAD)
    es = rows_by_target(y2024, *V2.MEAN_ES)
    cal = {k: v[V2.span_mask(y2024["origins"], *V2.CALIB)] for k, v in y2024.items()
           if k in ("x_raw", "y_kw", "origins", "targets")}
    rows = []
    for seed in V2.SEEDS:
        ck = OUT / f"site18_cqr_seed{seed}.pt"
        if ck.exists():
            model = CQRNet().to(V2.DEVICE)
            model.load_state_dict(torch.load(ck, weights_only=False)["state_dict"])
            model.eval()
        else:
            model = train_cqr(grad, es, sc, seed)
            torch.save({"state_dict": model.state_dict(), "seed": seed}, ck)
        qcal = quantiles(model, cal["x_raw"], sc)
        qhat = V2.conformal_q(np.sort(np.maximum(qcal[:, 0] - cal["y_kw"],
                                                 cal["y_kw"] - qcal[:, 2])))
        same = {k: v[V2.span_mask(y2024["origins"], *V2.SAME)] for k, v in y2024.items()
                if k in ("x_raw", "y_kw", "origins", "targets")}
        for tag, part in (("same", same), ("cross", y2025)):
            q = quantiles(model, part["x_raw"], sc)
            lo, up = q[:, 0] - qhat, q[:, 2] + qhat
            cov = (part["y_kw"] >= lo) & (part["y_kw"] <= up)
            isv = (up - lo) + 20 * np.maximum(lo - part["y_kw"], 0) + 20 * np.maximum(part["y_kw"] - up, 0)
            o = pd.DatetimeIndex(part["origins"])
            d1 = pd.Series(cov.astype(float)).groupby(o.normalize()).mean()
            rows.append({"method": "CQR", "seed": seed, "span": tag,
                         "PICP": float(cov.mean()), "PINAW": float(np.maximum(up - lo, 0).mean() / P18),
                         "IS": float(isv.mean()), "NIS": float(isv.mean() / P18),
                         "RCE1D": float(np.abs(d1 - 0.9).mean()), "qhat": float(qhat)})
        del model
        torch.cuda.empty_cache()
        print(f"seed {seed} done")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "site18_cqr_per_seed.csv", index=False)
    print(df.groupby(["method", "span"])[["PICP", "PINAW", "IS", "NIS", "RCE1D"]].median().round(4))


if __name__ == "__main__":
    main()
