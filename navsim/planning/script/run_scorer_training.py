"""Train the 9-net EPDMS component scorer (one single-output ScorerNet per component).

Each net regresses one EPDMS component -- focal-BCE for the 8 binary components,
MSE for EP (the one continuous ratio). At selection time the nine predictions are
stacked and EPDMS is composed via compose_epdms; the proposal argmax is the pick.

Training is EPOCH-MAJOR: each epoch makes one pass for all nine nets over the
same batches, then evaluates the COMPOSED selection on val. Checkpoints are
selected on realized EPDMS of the composed pick, not on val loss (val loss keeps
falling while realized degrades late in training). Each is the nine-net SET:
  comp{j}_bestval.pt     -- best realized EPDMS on the held-out val logs. Always
                            written; this is the reported checkpoint.
  comp{j}_bestreal.pt    -- best realized EPDMS under --select. With --select navtest
                            (all paper sweeps) it is selected ON NAVTEST and is a
                            diagnostic only, not held-out performance. With the
                            default --select val it is the same set as bestval.
  comp{j}_bestnavtest.pt -- --select val with --navtest_from > 0 only: the
                            navtest-selected set (diagnostic).
  comp{j}_last.pt        -- last epoch

Candidates are scored INDEPENDENTLY (ScorerNet masks cross-candidate attention),
so a candidate's score never depends on which others are in the set.

Deployed run: scripts/train_scorer.sh mf|dd plain|design|random [seeds...]
Result:       the closing 'done.' line of train.log: val EPDMS of comp*_bestval.pt, its
              epoch and its navtest EPDMS. scripts/eval_official.sh scores the MeanFuser
              scorers through the official evaluator.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from navsim.agents.meanfuser.scorer_net import ScorerNet, compose_epdms, EPDMS_NAMES
from navsim.planning.training.scorer_dataset import ScorerDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

EP_IDX = EPDMS_NAMES.index("EP")  # the only continuous component -> MSE
CK = os.path.join(os.environ.get("NAVSIM_EXP_ROOT", "exp"), "meanfuser_checkpoints")  # navtest defaults
FOCAL_GAMMA = 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True,
                   help="candidate dataset pkl: proposals + gt_trajectory per token (scripts/dump_candidates.sh "
                        "or tools/make_trainsets.py)")
    p.add_argument("--ctx", required=True, help="scene-context features pkl")
    p.add_argument("--labels", required=True, help="9-component EPDMS labels pkl (tools/dump_components.py)")
    p.add_argument("--out", required=True, help="checkpoint output directory")
    p.add_argument("--human", action="store_true",
                   help="add the HUMAN (logged GT) trajectory as an extra training "
                        "action/label, excluded from the re-rank argmax pool. Requires a "
                        "'human' key in --labels (scripts/label_slots.sh mf writes it).")
    p.add_argument("--logs", help="token->log_name pkl: hold out whole LOGS for val "
                                  "(avoids adjacent-frame leakage of a random-token split)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=0,
                   help="RNG seed for the train/val split; fixed so separate runs "
                        "share one held-out set and stay comparable")
    p.add_argument("--d_model", type=int, default=128, help="128 = MeanFuser, 256 = DiffusionDrive")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--cpu", action="store_true",
                   help="allow training on CPU (debug only; a real run is GPU-only)")
    p.add_argument("--val_pool", type=int, default=None,
                   help="val realized/oracle argmax over the FIRST N slots only (deployment "
                        "pool). Needed when later slots carry NaN epdms (free-label slots).")
    p.add_argument("--select", choices=["val", "navtest"], default="val",
                   help="which realized EPDMS the comp*_bestreal.pt checkpoint tracks. "
                        "'val' = held-out navtrain logs. 'navtest' = the TEST set: the "
                        "number it reports is optimistically biased and is NOT held-out "
                        "performance -- a diagnostic only. comp*_bestval.pt (val-selected, "
                        "the reported checkpoint) is written in both cases; under 'val' the "
                        "navtest-selected set is saved as comp*_bestnavtest.pt when "
                        "--navtest_from > 0.")
    p.add_argument("--navtest_from", type=int, default=0,
                   help="from this epoch on, report REALIZED NAVTEST EPDMS + all nine "
                        "components each epoch (0 = off; forced to 1 by --select navtest). "
                        "It never affects comp*_bestval.pt; it selects comp*_bestreal.pt under "
                        "--select navtest and comp*_bestnavtest.pt under --select val. "
                        "Adds ~1 s per epoch.")
    p.add_argument("--navtest_candidates", default=f"{CK}/plain_dataset_navtest/dataset.pkl")
    p.add_argument("--navtest_ctx", default=f"{CK}/plain_dataset_navtest/context_features.pkl")
    p.add_argument("--navtest_labels", default=f"{CK}/components_mf_navtest.pkl")
    args = p.parse_args()
    if args.select == "navtest" and args.navtest_from <= 0:
        args.navtest_from = 1
    assert args.val_frac > 0, "val split required (bestreal checkpointing needs it)"

    torch.manual_seed(args.seed)
    # Fail loudly instead of silently falling back to CPU: a full run is ~100 epochs
    # over 100k scenes, which is hours on GPU and effectively never on CPU. The usual
    # cause is a driver/library version mismatch after an nvidia package upgrade
    # (nvidia-smi says "Driver/library version mismatch") -- fixed by rebooting.
    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available -- refusing to train on CPU.\n"
            "  check : nvidia-smi   and   cat /proc/driver/nvidia/version\n"
            "  if the loaded kernel module version != the installed libnvidia-ml "
            "version, reboot to load the upgraded module.\n"
            "  to train on CPU anyway (debug only), pass --cpu")
    device = torch.device("cpu" if args.cpu else "cuda")
    logger.info(f"device = {device}"
                + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    full = ScorerDataset(args.candidates, args.ctx, args.labels, include_human=args.human)
    num_proposals = full.num_proposals
    if args.val_pool:
        num_proposals = min(args.val_pool, num_proposals)

    # split RNG independent of the init seed -> every run shares ONE held-out
    # set (so no run ever trains on another's val logs).
    split_rng = np.random.RandomState(args.split_seed)
    if args.logs:
        # LOG-level holdout: scenes within a log are temporally adjacent and highly
        # correlated, so a random-token val set leaks into training.
        import pickle
        tok2log = pickle.load(open(args.logs, "rb"))
        logs = sorted(set(tok2log[t] for t in full.tokens if t in tok2log))
        split_rng.shuffle(logs)
        val_logs = set(logs[: int(len(logs) * args.val_frac)])
        val_tokens = [t for t in full.tokens if tok2log.get(t) in val_logs]
        train_tokens = [t for t in full.tokens if tok2log.get(t) not in val_logs]
        logger.info(f"log-level val split: {len(val_logs)}/{len(logs)} logs held out "
                    f"({len(val_tokens)} val scenes)")
    else:
        shuffled = list(full.tokens); split_rng.shuffle(shuffled)
        n_val = int(len(shuffled) * args.val_frac)
        val_tokens, train_tokens = shuffled[:n_val], shuffled[n_val:]
    tr = full.subset(train_tokens)
    va = full.subset(val_tokens)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=4)
    logger.info(f"train {len(train_tokens)} val {len(val_tokens)} "
                f"proposals/scene {num_proposals}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    NC = len(EPDMS_NAMES)
    models = [ScorerNet(d_model=args.d_model, dropout=args.dropout).to(device) for _ in range(NC)]
    opts = [torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd) for m in models]
    best_real = -1.0
    best_alt = -1.0
    best_val_ep, best_val_nt = -1, None   # epoch of comp*_bestval.pt and its navtest EPDMS (closing line)

    # ---- optional per-epoch NAVTEST reporting. It selects only comp*_bestreal.pt
    # (--select navtest) or comp*_bestnavtest.pt (--select val); never comp*_bestval.pt ----
    nt = None
    if args.navtest_from > 0:
        import pickle
        n_rl = pickle.load(open(args.navtest_candidates, "rb"))
        n_ctx = pickle.load(open(args.navtest_ctx, "rb"))
        n_blob = pickle.load(open(args.navtest_labels, "rb")); n_lab = n_blob["data"]
        n_tok = sorted(set(n_rl) & set(n_ctx) & set(n_lab))
        # DiffusionDrive's raw dump calls the candidate set pred_trajectorys_array
        _pk = "proposals" if "proposals" in n_rl[n_tok[0]] else "pred_trajectorys_array"
        # deployment pool width: follow --val_pool, else the file's own width.
        # (Was hardcoded to 8, which silently truncated DiffusionDrive's 20.)
        NP = args.val_pool or int(np.asarray(n_rl[n_tok[0]][_pk]).shape[0])
        nt = {
            "A": torch.from_numpy(np.stack([np.asarray(n_rl[t][_pk])[:NP] for t in n_tok])
                                  .astype(np.float32)).to(device),
            "X": torch.from_numpy(np.stack([n_ctx[t] for t in n_tok]).astype(np.float32)).to(device),
            "C": np.stack([np.asarray(n_lab[t]["components"])[:NP] for t in n_tok]).astype(np.float64),
            "E": np.stack([np.asarray(n_lab[t]["epdms"])[:NP] for t in n_tok]).astype(np.float64),
        }
        logger.info(f"navtest reporting from epoch {args.navtest_from}: "
                    f"{len(n_tok)} scenes x {NP} proposals "
                    f"(oracle {nt['E'].max(1).mean():.4f})")

    def navtest_report():
        """Realized navtest EPDMS + per-component realized of the composed argmax pick."""
        for m in models:
            m.eval()
        S = nt["X"].shape[0]
        P = torch.zeros(S, nt["A"].shape[1], NC)
        with torch.no_grad():
            for i in range(0, S, 2048):
                cc = nt["X"][i:i + 2048]; aa = nt["A"][i:i + 2048]
                P[i:i + cc.shape[0]] = torch.stack([m(cc, aa) for m in models], -1).cpu()
        pk = compose_epdms(P).numpy().argmax(1)
        ar = np.arange(S)
        return nt["E"][ar, pk].mean(), [np.nanmean(nt["C"][ar, pk, j]) for j in range(NC)]

    def comp_loss(pred, tgt, j):
        """Per-component loss, NaN-safe (TFC is NaN on the first frame of a log ->
        those targets are masked out, matching the scorer). EP (continuous) -> MSE;
        the 8 binary components -> focal-BCE (emphasizes rare high-leverage failures)."""
        mask = ~torch.isnan(tgt)
        if not mask.any():
            return (pred * 0.0).sum(), 0
        pred, tgt = pred[mask], tgt[mask]; n = int(mask.sum())
        if j == EP_IDX:
            per = (pred - tgt) ** 2
        else:
            pr = pred.clamp(1e-6, 1 - 1e-6); t = tgt.clamp(0.0, 1.0)
            pt = pr * t + (1 - pr) * (1 - t)  # prob of the true class
            per = -((1 - pt) ** FOCAL_GAMMA) * torch.log(pt)
        return per.mean(), n

    def joint_val():
        """Composed selection on val: realized EPDMS / top-1, plus per-comp val loss."""
        for m in models:
            m.eval()
        vloss = [0.0] * NC; vcnt = [0] * NC
        top1, realized, oracle, n = 0, 0.0, 0.0, 0
        with torch.no_grad():
            for b in vl:
                c = b["context"].to(device); a = b["actions"].to(device)
                comp = b["components"].to(device); rew = b["rewards"].to(device)
                preds = torch.stack([models[j](c, a) for j in range(NC)], dim=-1)
                for j in range(NC):
                    lf, cnt = comp_loss(preds[..., j], comp[..., j], j)
                    vloss[j] += lf.item() * cnt; vcnt[j] += cnt
                score = compose_epdms(preds)[:, :num_proposals]
                r_p = rew[:, :num_proposals]
                pk = score.argmax(1)  # the re-rank pick
                top1 += (pk == r_p.argmax(1)).sum().item()
                realized += r_p.gather(1, pk[:, None]).sum().item()
                oracle += r_p.max(1).values.sum().item(); n += rew.shape[0]
        return ([v / max(c, 1) for v, c in zip(vloss, vcnt)], top1 / n,
                realized / n, oracle / n)

    for ep in range(args.epochs):
        tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=4)
        for m in models:
            m.train()
        tr_sum = [0.0] * NC; tr_cnt = [0] * NC
        for b in tl:
            c = b["context"].to(device); a = b["actions"].to(device)
            comp = b["components"].to(device)
            total = c.new_zeros(())
            for j in range(NC):
                loss, cnt = comp_loss(models[j](c, a), comp[..., j], j)
                total = total + loss
                tr_sum[j] += loss.item() * cnt; tr_cnt[j] += cnt
            for o in opts:
                o.zero_grad()
            total.backward()
            for o in opts:
                o.step()
        for j in range(NC):
            torch.save(models[j].state_dict(), out / f"comp{j}_last.pt")

        vloss, top1, realized, oracle = joint_val()
        tr_str = " ".join(f"{nm}={tr_sum[j]/max(tr_cnt[j],1):.4f}" for j, nm in enumerate(EPDMS_NAMES))
        vl_str = " ".join(f"{nm}={vloss[j]:.4f}" for j, nm in enumerate(EPDMS_NAMES))
        # navtest is evaluated BEFORE selection so it can drive it when asked
        nt_ep, nt_str = None, ""
        if nt is not None and ep + 1 >= args.navtest_from:
            nt_ep, nt_comp = navtest_report()
            nt_str = ("  || NAVTEST " + f"{nt_ep:.4f} ["
                      + " ".join(f"{nm}={v:.4f}" for nm, v in zip(EPDMS_NAMES, nt_comp)) + "]")

        # `bestreal` follows --select; the other criterion is saved as comp*_best<alt>.pt.
        # comp*_bestval.pt (val-selected, the reported checkpoint) is written either way:
        # as the alt under --select navtest, as a copy of bestreal under --select val.
        sel = nt_ep if args.select == "navtest" else realized
        alt = realized if args.select == "navtest" else nt_ep
        alt_tag = "val" if args.select == "navtest" else "navtest"
        mark = ""
        if sel is not None and sel > best_real:
            best_real = sel
            for j in range(NC):
                torch.save(models[j].state_dict(), out / f"comp{j}_bestreal.pt")
                if args.select == "val":
                    torch.save(models[j].state_dict(), out / f"comp{j}_bestval.pt")
            if args.select == "val":
                best_val_ep, best_val_nt = ep, nt_ep
            mark = f"  <- new best {args.select}"
        if alt is not None and alt > best_alt:
            best_alt = alt
            for j in range(NC):
                torch.save(models[j].state_dict(), out / f"comp{j}_best{alt_tag}.pt")
            if args.select == "navtest":
                best_val_ep, best_val_nt = ep, nt_ep

        logger.info(f"epoch {ep:03d} | train[{tr_str}] | val[{vl_str}] "
                    f"| top1 {top1:.4f} | realized {realized:.4f} (best {best_real:.4f}, "
                    f"oracle {oracle:.4f}){mark}{nt_str}")

    best_val = best_real if args.select == "val" else best_alt
    if args.select == "navtest":
        rest = (f" | best navtest {best_real:.4f} -> comp*_bestreal.pt "
                f"(navtest-selected, diagnostic only)")
    else:
        rest = " | comp*_bestreal.pt = comp*_bestval.pt" + (
            f" | best navtest {best_alt:.4f} -> comp*_bestnavtest.pt (diagnostic only)"
            if best_alt > 0 else "")
    nt_note = f" | its navtest EPDMS {best_val_nt:.4f}" if best_val_nt is not None else ""
    logger.info(f"done. best val realized EPDMS {best_val:.4f} at epoch {best_val_ep:03d} -> comp*_bestval.pt "
                f"(validation-selected, the reported checkpoint){nt_note}{rest}")
    if args.select == "navtest":
        logger.warning("comp*_bestreal.pt was selected ON NAVTEST -- its number is not "
                       "held-out performance; evaluate comp*_bestval.pt (--tag bestval)")


if __name__ == "__main__":
    main()
