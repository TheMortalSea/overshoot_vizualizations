"""Vegetation carbon (cVeg) charts: one chart per model, with all of the model's scenarios.

Built on wiemip_charts.py (keep both in the same folder): RECCAP mask, grid weighting,
config, caching and path handling come from there.

Values are regional totals in Pg C (RECCAP permafrost region by default, --region global
for the globe). A model is charted only if the registry has cVeg for it.

Scenarios and colours
---------------------
--scenarios all uses every future scenario in the registry (wiemip_registry.const.SCENARIO_COLORS,
minus hist/ctrl). A model only gets the scenarios it actually has. Lines use the registry's
scenario colours, ordered as in the registry (vl ... hl).

History (--history)
-------------------
Also computes the historical run (--hist-sim, default "hist") for every model that has it and
draws it in the registry's "hist" colour ahead of the scenarios. Do not put "hist" in --scenarios.

Notebook use:
    import wiemip_veg_charts as wvc
    wvc.main(["--io-module", "wiemip_io", "--forcing", "gfdl", "--scenarios", "all",
              "--history", "--check-only"])
    wvc.main(["--io-module", "wiemip_io", "--forcing", "gfdl", "--scenarios", "all",
              "--history", "--mask-regions", "101", "103", "106", "108", "--share-y"])
"""

from __future__ import annotations

import argparse
import importlib
import sys
import traceback
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:               # pasted into a notebook cell
    sys.path.insert(0, str(Path.cwd()))
import wiemip_charts as wc

try:
    from wiemip_registry.const import SCENARIO_COLORS
except Exception:               # registry too old / not importable
    SCENARIO_COLORS = {}

VAR = "cVeg"
TITLE = "Vegetation carbon"


def scen_colour(s: str) -> str:
    """Registry colour; a scenario missing from the registry (e.g. hl_cf) borrows its base scenario's colour."""
    if s in SCENARIO_COLORS:
        return SCENARIO_COLORS[s]
    base = s[:-3] if s.endswith("_cf") else s
    return SCENARIO_COLORS.get(base, "#555555")


def register_charts():
    """Register cVeg in the shared chart table (regional total, Pg C).

    Called at import AND at the start of main(), so reloading wiemip_charts (which replaces
    wc.CHARTS with a fresh dict) cannot leave the key missing.
    """
    wc.CHARTS[VAR] = dict(variable=VAR, kind="total", title=TITLE, ylabel=f"{TITLE} (Pg C)")


register_charts()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_model(io, cfg, model, hist_sim=None):
    """Which scenarios (and whether history) have cVeg for this model."""
    scen, hist = [], False
    for s in cfg.scenarios:
        try:
            io.retrieve(VAR, model, cfg.forcing, s, cfg.factorial)
            scen.append(s)
        except Exception:
            pass
    if hist_sim:
        try:
            io.retrieve(VAR, model, cfg.forcing, hist_sim, cfg.factorial)
            hist = True
        except Exception:
            pass
    return dict(scen=scen, hist=hist)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scenario_styles(scenarios) -> dict:
    """Registry colour per scenario. Counterfactuals the registry doesn't list are dashed."""
    return {s: dict(color=scen_colour(s),
                    ls="--" if (s.endswith("_cf") and s not in SCENARIO_COLORS) else "-")
            for s in scenarios}


def plot_model(io, cfg, model, series, hist, styles, ylim):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    order = {s: i for i, s in enumerate(SCENARIO_COLORS)}
    if hist is not None:
        ax.plot(hist.index, hist.values, lw=1.8, color=scen_colour("hist"), label="HIST")
    for s in sorted((s for s in cfg.scenarios if s in series), key=lambda s: order.get(s, 99)):
        ser = series[s]
        ax.plot(ser.index, ser.values, lw=2.4 if s.startswith("ml") else 1.8,
                label=s.upper(), **styles[s])
    if hist is not None and series:
        ax.axvline(min(int(s.index.min()) for s in series.values()), color="0.6", ls=":", lw=1)
    if ylim:
        ax.set_ylim(*ylim)
    region = "RECCAP2 permafrost region" if cfg.region == "reccap" else "global"
    ax.set_title(f"{model} - {TITLE} - {region} ({cfg.forcing.upper()})", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{TITLE} (Pg C)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2 if len(series) > 4 else 1, fontsize=9, title="Scenario")
    fig.tight_layout()

    out_dir = io.FIGURE_DIR / f"vegc_{cfg.forcing}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ("_" + "-".join(str(int(r)) for r in cfg.mask_regions)
           if cfg.mask_regions and cfg.region == "reccap" else "")
    h = "_hist" if hist is not None else ""
    path = out_dir / f"{model}_cVeg_{cfg.region}{ids}{h}_{date.today():%Y-%m-%d}.png"
    fig.savefig(path, dpi=200)
    print(f"  saved {path.name}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--io-module", default="wiemip_io")
    ap.add_argument("--models", nargs="+", default=None, help="default: every registered model")
    ap.add_argument("--forcing", default="ukesm")
    ap.add_argument("--scenarios", nargs="+", default=list(wc.DEFAULT_SCENARIOS),
                    help="scenarios to look for, or 'all' for every future scenario in the registry")
    ap.add_argument("--region", default="reccap", choices=["reccap", "global"])
    ap.add_argument("--mask", default=wc.DEFAULT_MASK)
    ap.add_argument("--mask-var", default=None)
    ap.add_argument("--mask-regions", nargs="+", type=float, default=None)
    ap.add_argument("--share-y", action="store_true", help="share the y range across models")
    ap.add_argument("--history", action="store_true", help="also compute and draw the historical run")
    ap.add_argument("--hist-sim", default="hist", help="registry simulation name for the historical run")
    ap.add_argument("--factorial", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached series")
    ap.add_argument("--check-only", action="store_true", help="show the plan for each model; load no data")
    args = ap.parse_args(argv)

    register_charts()          # wiemip_charts may have been reloaded since import

    here = Path(globals().get("__file__", Path.cwd())).resolve().parent
    for p in (here, here.parent, here.parent / "src", Path.cwd()):
        sys.path.insert(0, str(p))
    io = importlib.import_module(args.io_module)

    hist_sim = args.hist_sim if args.history else None
    if args.scenarios == ["all"]:
        scenarios = [s for s in SCENARIO_COLORS if not s.startswith(("hist", "ctrl"))]
        if not scenarios:
            scenarios = list(getattr(io.wr, "overshoot_simulations", wc.DEFAULT_SCENARIOS))
    else:
        scenarios = list(args.scenarios)
    scenarios = [s for s in scenarios if s != hist_sim]

    models = args.models or list(getattr(io.wr, "models", ()))
    if not models:
        sys.exit("No models: pass --models.")
    cfg = wc.ChartConfig(models=tuple(models), forcing=args.forcing, scenarios=tuple(scenarios),
                         region=args.region, mask_var=args.mask_var,
                         mask_regions=tuple(args.mask_regions) if args.mask_regions else None,
                         factorial=args.factorial, refresh=args.refresh)

    print(f"Vegetation carbon plan ({cfg.forcing.upper()}):")
    plans, todo = {}, []
    for m in cfg.models:
        plans[m] = plan_model(io, cfg, m, hist_sim)
        p = plans[m]
        hs = f"; {hist_sim}: {'yes' if p['hist'] else 'no'}" if hist_sim else ""
        print(f"  {m:<22} cVeg scenarios: {', '.join(p['scen']) or 'none'}{hs}")
        if p["scen"]:
            todo.append(m)
    print(f"\nCharts: {len(todo)} ({', '.join(todo) or 'none'})")
    results = {"plans": plans, "series": {}, "hist": {}, "figures": {}}
    if args.check_only or not todo:
        return results

    mask = None
    if cfg.region == "reccap":
        mpath = wc.resolve_path(io, args.mask)
        mask = wc.load_region_mask(mpath, cfg.mask_var, cfg.mask_regions)
        print(f"\nMask {mpath.name}: {mask.description}")

    print("\nComputing...")
    wc_announced: set = set()
    data: dict = {}            # model -> {scenario: Series}
    hist_data: dict = {}       # model -> Series
    errors: dict = {}
    shown_tb = False

    def record(exc, label):
        """Collect an error; print the full traceback for the first one only."""
        nonlocal shown_tb
        if not shown_tb:
            shown_tb = True
            print(f"  first failure ({label}), full traceback:")
            traceback.print_exc(file=sys.stdout)
        errors.setdefault(f"{type(exc).__name__}: {str(exc)[:130]}", []).append(label)

    for m in todo:
        for s in plans[m]["scen"]:
            try:
                data.setdefault(m, {})[s] = wc.get_series(io, cfg, mask, m, s, VAR, wc_announced)
            except Exception as e:
                record(e, f"{m}/{s}")
        if hist_sim and plans[m]["hist"]:
            try:
                hist_data[m] = wc.get_series(io, cfg, mask, m, hist_sim, VAR, wc_announced)
            except Exception as e:
                record(e, f"{m}/{hist_sim}")
    for msg, runs in errors.items():
        print(f"  FAILED ({len(runs)}): {', '.join(runs[:8])}{' ...' if len(runs) > 8 else ''} -> {msg}")
    results["series"], results["hist"] = data, hist_data

    print("\nCharts:")
    styles = scenario_styles(scenarios)
    ylim = None
    if args.share_y:
        vals = [v for d in data.values() for s in d.values() for v in (s.min(), s.max())]
        vals += [v for s in hist_data.values() for v in (s.min(), s.max())]
        if vals:
            pad = 0.05 * ((max(vals) - min(vals)) or 1.0)
            ylim = (max(0.0, min(vals) - pad), max(vals) + pad)
    for m, series in data.items():
        results["figures"][m] = plot_model(io, cfg, m, series, hist_data.get(m), styles, ylim)
    return results


if __name__ == "__main__":
    main()