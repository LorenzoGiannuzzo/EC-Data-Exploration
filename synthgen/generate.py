"""Stage 4 — Generation.

Draws synthetic points of delivery of a requested typology and writes a year of
quarter-hourly or hourly consumption for each.

How a year is produced. The calendar is built in local civil time, so the two
daylight saving days come out with 92 and 100 quarter-hours of their own accord
and the timestamps carry the truth rather than a convention. For every day a
regime is drawn from the daily chain, closed, low, medium or high, conditioned on
the season and the day type of that day; the regime then selects which
quarter-hourly transition matrix governs it. Inside the day the chain walks from
one level to the next, carrying its momentum, and the walk continues across
midnight rather than restarting, so the night is continuous. The level is finally
turned back into kilowatt-hours by inverting the empirical distribution of the
bin, at a position that follows an AR(1): a point sitting high in its bin tends to
stay high, which is what keeps the curve from looking like a staircase.

Every generated point draws a size anchor from the empirical distribution of the
real points of its stratum, and the year is placed at that size with a single
multiplicative factor. The alternative, leaving every generated point at the mean
of its stratum, produces n curves that differ only by noise and share one
magnitude, which is not what a network sees. The factor applies to the whole year
at once, so the dynamics the chain produced are untouched.

Outputs, in the directory given
    <typology>_<nnn>.csv       one file per generated point
    generation_manifest.csv    what was drawn, and from which model
    validation/*.png           only when validation is asked for

Run
    python -m synthgen.generate --typology 47 --ateco-level 1 --n 50 \\
        --resolution 15min --year 2025 --outdir results/synthetic --seed 42 \\
        --validation
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import DAYTYPES, annotate
from .config import load_config
from .estimate import to_state
from .taxonomy import normalise_typology

TZ = "Europe/Rome"
#Lorenzo Giannuzzo: a point with fewer valid days than this cannot speak for itself
MIN_DAYS_PERSONAL = 60


#Lorenzo Giannuzzo: ── model selection ──────────────────────────────────────────────────────────
def load_manifest(models_dir: Path) -> pd.DataFrame:
    path = models_dir / "manifest.csv"
    if not path.exists():
        raise SystemExit(
            f"\n  no model found in {models_dir}. Run:\n"
            f"      python -m synthgen.estimate --level 1 --all\n")
    return pd.read_csv(path)


def pick_models(manifest: pd.DataFrame, typology: str, level: int,
                size_range: tuple[float, float] | None) -> pd.DataFrame:
    """The strata a request resolves to, with a reason when it resolves to none."""
    sel = manifest[(manifest["level"] == level)
                   & (manifest["typology"].astype(str) == typology)]
    if not len(sel):
        available = sorted(manifest.loc[manifest["level"] == level, "typology"]
                           .astype(str).unique())
        raise SystemExit(
            f"\n  no model for typology {typology!r} at level {level}.\n"
            f"  available: {', '.join(available)}\n")
    if size_range is not None:
        lo, hi = float(size_range[0]), float(size_range[1])
        by_energy = (sel["stratified_on"] == "annual_kWh").any()
        if by_energy:
            print(f"  ! this typology is stratified on annual energy, not on "
                  f"contractual power, because nearly every point declares the "
                  f"same power. The size range is matched on the power actually "
                  f"observed in each stratum, which overlaps between them.")
        overlap = sel[(sel["power_kW_max"] >= lo) & (sel["power_kW_min"] <= hi)]
        if not len(overlap):
            spans = ", ".join(f"{r.key} [{r.power_kW_min:g}, {r.power_kW_max:g}] kW"
                              for r in sel.itertuples())
            raise SystemExit(
                f"\n  no stratum of {typology!r} overlaps {lo:g}-{hi:g} kW.\n"
                f"  strata available: {spans}\n")
        sel = overlap
    return sel


def load_model(models_dir: Path, key: str) -> dict:
    z = np.load(models_dir / f"{key}.npz", allow_pickle=True)
    m = {k: z[k] for k in z.files}
    m["meta"] = json.loads(str(m["meta"][0]))
    m["seasons"] = [str(s) for s in m["seasons"]]
    m["daytypes"] = [str(d) for d in m["daytypes"]]
    return m


#Lorenzo Giannuzzo: ── the signature of one real point ──────────────────────────────────────────
def pod_counts_for(model: dict, pod: str, days_all: pd.DataFrame,
                   curves_all) -> np.ndarray | None:
    """How often one real point moved from each level, at each hour of the day.

    A stratum matrix is the average of the points in it, and the average of a shop
    that opens at eight and of a load that never switches off resembles neither:
    that is why the generated load factor has no tail above 0.6 and the curves
    have no square fronts. These counts are the schedule of one point, pooled over
    regime, season and day type, which five hundred days cannot resolve, and kept
    by hour of the day and by level, which is what makes that point itself. They
    are mixed into every step of the walk with the usual n / (n + k) weight, so a
    level the point visited often speaks for itself and one it never visited falls
    back on its stratum.

    Returned indexed (block, from level, to level), or None when the point is too
    thin to say anything.
    """
    mask = (days_all["valid"].to_numpy()
            & (days_all["pod"].to_numpy().astype(str) == str(pod)))
    if int(mask.sum()) < MIN_DAYS_PERSONAL:
        return None
    cur = np.asarray(curves_all[mask], dtype="float64")
    st = to_state(cur, model["edges"])
    n_blocks = int(model["n_blocks"][0])
    n_states = model["trans"].shape[-1]
    blocks = (np.arange(cur.shape[1]) * n_blocks // cur.shape[1]).astype(int)
    counts = np.zeros((n_blocks, n_states, n_states))
    src, dst = st[:, :-1], st[:, 1:]
    ok = (src >= 0) & (dst >= 0)
    rows, cols = np.nonzero(ok)
    np.add.at(counts, (blocks[cols + 1], src[rows, cols], dst[rows, cols]), 1.0)
    return counts


#Lorenzo Giannuzzo: ── the walk ─────────────────────────────────────────────────────────────────
def year_calendar(year: int, seasons: dict) -> pd.DataFrame:
    """Quarter-hourly civil calendar of the year, in local time.

    Built tz aware, so the March day holds 92 slots and the October one 100
    without any special casing: the clock, not the code, decides how long a civil
    day is.
    """
    idx = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:45",
                        freq="15min", tz=TZ)
    df = pd.DataFrame({"timestamp": idx})
    naive = idx.tz_localize(None)
    df["date"] = pd.to_datetime(naive.date)
    day = annotate(df[["date"]].drop_duplicates().reset_index(drop=True), seasons)
    return df.merge(day, on="date", how="left")


def walk(model: dict, cal: pd.DataFrame, rng: np.random.Generator,
         shift: int = 0, pod_counts: np.ndarray | None = None,
         mix_k: float = 30.0, regularity: float = 1.0,
         identity: float = 0.0, closure_class: int = 0) -> np.ndarray:
    """One synthetic year, in kWh per quarter-hour, at the scale of the stratum.

    `regularity` is the exponent every transition row is raised to before it is
    drawn from, held for the whole year, and it is what makes one generated point
    a creature of habit and another an erratic one. At one the row is the
    estimated one. Above one the row concentrates on its likeliest destination,
    so the point opens at the same hour every morning, holds the level it reached
    instead of leaving it after a quarter, and its Monday resembles its Tuesday.

    The chain estimated on a stratum reproduces the statistics of that stratum
    and not the habit of any point in it, which is a different thing and the
    figures separate them: a mean day can be right to within a total variation of
    0.06 while the distance between one generated day and the next is 0.49 where
    the metered points sit at 0.17, and while the autocorrelation at twenty four
    hours is 0.21 against their 0.58. Sharpening the rows is the cheapest way to
    buy that habit back without giving the chain a dictionary of shapes to copy,
    which is the one thing this model is not allowed to have.

    `identity` is the second half of the same idea, applied not to which level
    the point moves to but to where inside that level it sits. The emission of a
    bin is the distribution of what every point of the stratum was read at while
    it was in that bin, so its low end belongs to the small points and its high
    end to the large ones. Letting the position wander over the whole of it, as
    an AR(1) alone does, gives a point that is small at nine in the morning and
    large at ten, which is what turns a plateau into a noisy hill and costs the
    load factor: on a hand built model the same walk moves from 0.25 to 0.36 as
    the position is held still, while the sharpening of the rows leaves it
    untouched to three decimals. At zero the behaviour is the one estimated. At
    w the point draws its own place in the bin once and keeps it, and only the
    remaining 1 - w squared of the variance is left to the walk, so the marginal
    distribution over the whole generated population is the estimated one and
    what changes is how that variance is split between the points and the hours.
    """
    trans = model["trans"]
    #Lorenzo Giannuzzo: The daily chain of the point's own closure class. How readily a point shuts
    # for a day belongs to the point and not to the day, so the class is taken
    # from the real point this profile is anchored to and held for the year.
    # Models estimated before the class existed carry a single pooled chain and
    # are read as they were.
    r_trans = model["regime_trans"]
    if r_trans.ndim == 5:
        r_trans = r_trans[int(np.clip(closure_class, 0, r_trans.shape[0] - 1))]
    emission, rho = model["emission"], float(model["rho"][0])
    n_blocks = int(model["n_blocks"][0])
    n_states = trans.shape[-1]
    n_qt = emission.shape[-1]
    grid = np.linspace(0.0, 1.0, n_qt)

    s_of = {s: i for i, s in enumerate(model["seasons"])}
    d_of = {d: i for i, d in enumerate(model["daytypes"])}

    n_ctx = int(model["n_ctx"][0]) if "n_ctx" in model else 3
    flat_long = int(model["flat_long"][0]) if "flat_long" in model else 4
    sharp = float(regularity)

    def draw(row: np.ndarray, here: int) -> int:
        """One destination, from a row sharpened off the diagonal only.

        The probability of staying where it is is left exactly as estimated, and
        the exponent is applied to the rest, which is then renormalised back onto
        the mass that was there before. So the point keeps the dwell the data gave
        it and only becomes more decided about where it goes when it goes
        somewhere, which is what a habit is.

        Sharpening the whole row instead was tried and is a trap. At a quarter of
        an hour the likeliest destination from any level is that level, so the
        exponent falls on the diagonal and the chain stops moving rather than
        acquiring a schedule. It scores beautifully on the two checks that a
        frozen series passes by definition, day to day repeatability at 0.163
        against a metered 0.174 and autocorrelation at twenty four hours at 0.562
        against 0.580, while the share of the year spent at zero goes from 0.03 to
        0.36, the mean day flattens into a line, and one profile in fifty produces
        a thousandth of the energy of the point it was anchored to.

        State zero is held out along with the diagonal, and for the same reason
        read one step further. Zero is not a level among the others, it is the
        point having stopped, and how often a point stops is something the data
        say and the exponent has no business revising. Sharpening across it makes
        the walk decided about going to zero and then, the diagonal being what it
        is, about staying there: on the domestic typology the exponent bought a
        load factor of 0.218 against a metered 0.211, and paid for it with a year
        spent at zero of 0.382 against a metered 0.004 and a median daily energy
        at a quarter of the right one. Sharpened among the positive levels only,
        the walk still settles onto the plateau it reaches instead of crossing it,
        which is where the load factor was won, while the probability of shutting
        stays exactly as estimated.

        Raising to the exponent leaves a zero at zero, so a destination the
        stratum never reached stays unreachable however regular the point is.
        """
        if sharp == 1.0:
            return int(rng.choice(n_states, p=row))
        held = {here, 0}                      #Lorenzo Giannuzzo: the diagonal, and the closed state
        off = row.copy()
        for h in held:
            off[h] = 0.0
        moving = off.sum()
        if moving <= 0.0:
            return int(rng.choice(n_states, p=row))
        off = np.power(off, sharp)
        total = off.sum()
        if total <= 0.0:
            return int(rng.choice(n_states, p=row))
        off *= moving / total
        for h in held:
            off[h] = float(row[h])
        return int(rng.choice(n_states, p=off / off.sum()))

    out = np.zeros(len(cal), dtype="float64")
    state, ctx, dwell = 0, 2, 0
    x = float(rng.standard_normal())            #Lorenzo Giannuzzo: AR(1) on the copula scale
    #Lorenzo Giannuzzo: The point's own place inside a bin, drawn once and held for the year, and
    # the share of the variance left to the walk around it. The two add to one.
    w = float(np.clip(identity, 0.0, 0.999))
    own_level = w * float(rng.standard_normal()) if w > 0.0 else 0.0
    wander = math.sqrt(max(1.0 - w * w, 0.0))
    regime = int(rng.integers(0, r_trans.shape[-1]))

    for _, day in cal.groupby("date", sort=True):
        pos = day.index.to_numpy()
        n_q = len(pos)
        si = s_of.get(str(day["season"].iloc[0]), 0)
        di = d_of.get(str(day["daytype"].iloc[0]), 0)
        regime = int(rng.choice(r_trans.shape[-1], p=r_trans[si, di, regime]))
        #Lorenzo Giannuzzo: the shift is the point's own clock, held for the whole year
        blocks = (((np.arange(n_q) + shift) % n_q) * n_blocks // n_q).astype(int)

        for j in range(n_q):
            b = blocks[j]
            if j > 0 or pos[0] > 0:
                row = trans[regime, si, di, b, state * n_ctx + ctx]
                if pod_counts is not None:
                    #Lorenzo Giannuzzo: what this particular point does at this hour, from this
                    # level, weighted by how often it was seen doing it
                    c = pod_counts[b, state]
                    tot = float(c.sum())
                    if tot > 0:
                        w = tot / (tot + mix_k)
                        row = w * (c / tot) + (1.0 - w) * row
                        row = row / row.sum()
                nxt = draw(row, state)
                if nxt == state:
                    dwell += 1
                    ctx = 3 if dwell >= flat_long else 2
                else:
                    ctx = 1 if nxt > state else 0
                    dwell = 0
                state = nxt
            if state == 0:
                out[pos[j]] = 0.0
                continue
            x = rho * x + np.sqrt(max(1.0 - rho * rho, 1e-12)) * rng.standard_normal()
            z = own_level + wander * x
            u = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            out[pos[j]] = float(np.interp(u, grid, emission[state, b]))
    return out


def to_hourly(values: np.ndarray, cal: pd.DataFrame) -> pd.DataFrame:
    """Hourly energy is the sum of its quarters, never an interpolation."""
    df = pd.DataFrame({"timestamp": cal["timestamp"], "kWh": values})
    return (df.set_index("timestamp").resample("1h").sum().reset_index())


#Lorenzo Giannuzzo: ── entry point ──────────────────────────────────────────────────────────────
def generate(typology: str, ateco_level: int = 1,
             size_range: tuple[float, float] | None = None,
             n_profiles: int = 10, resolution: str = "15min",
             year: int = 2025, outdir: str | Path = "results/synthetic",
             seed: int = 42, validation: bool = False,
             models_dir: str | Path | None = None) -> pd.DataFrame:
    if resolution not in ("15min", "1h"):
        raise ValueError("resolution must be '15min' or '1h'")
    if n_profiles < 1:
        raise ValueError("n_profiles must be at least 1")

    cfg = load_config()
    models = Path(models_dir) if models_dir else cfg.models_dir
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    key_typ = normalise_typology(typology, ateco_level)

    power_margin = float(cfg.get("preprocessing.power_margin", 1.5))
    max_shift = int(cfg.get("generation.profile_shift_quarters", 0))
    mix_k = float(cfg.get("generation.personal_shrinkage",
                          cfg.get("estimation.shrinkage", 30)))
    personal = bool(cfg.get("generation.personal_chain", False))
    def regularity_range(typ: str) -> tuple[float, float, str]:
        """The exponent range for one typology, and where it came from.

        A default with a table of exceptions, rather than a formula keyed on the
        share of zero days. The effect of the exponent does depend on that share,
        and strongly: on typology 47, which draws nothing on 3 percent of its
        days, the range [1, 4] brings the load factor onto the metered one and
        costs about ten per cent of the median daily energy, and is worth it. On
        typology 01, which draws nothing on 31 percent of them, the same range
        costs three quarters of that energy and doubles the share of the year the
        generated points spend at zero, and is not. But two typologies are two
        points, and a curve fitted through two points and buried in the code is
        a guess nobody will ever revisit. An explicit table can be read, argued
        with, and corrected one line at a time.
        """
        table = cfg.get("generation.regularity_by_typology", {}) or {}
        for raw, value in table.items():
            name = str(raw).strip()
            if name == typ or name.zfill(2) == typ:
                return float(value[0]), float(value[-1]), "typology override"
        default = cfg.get("generation.regularity", [1.0, 1.0]) or [1.0, 1.0]
        return float(default[0]), float(default[-1]), "default"

    reg_lo, reg_hi, reg_source = regularity_range(
        normalise_typology(typology, ateco_level))
    if reg_lo < 1.0:
        raise ValueError("generation.regularity must start at 1.0 or above")
    identity = float(cfg.get("generation.level_identity", 0.0) or 0.0)
    if not 0.0 <= identity < 1.0:
        raise ValueError("generation.level_identity must lie in [0, 1)")
    days_all = curves_all = None
    if personal:
        days_all = pd.read_parquet(cfg.cache_dir / "days.parquet")
        curves_all = np.load(cfg.cache_dir / "curves.npy", mmap_mode="r")
    manifest = load_manifest(models)
    chosen = pick_models(manifest, key_typ, ateco_level, size_range)
    rng = np.random.default_rng(seed)
    cal = year_calendar(year, cfg["preprocessing"]["seasons"])

    print(f"\n{'=' * 78}\nSTAGE 4 — GENERATION\n{'=' * 78}\n")
    print(f"  typology {key_typ} at level {ateco_level}, {n_profiles} profiles, "
          f"{resolution}, year {year}, seed {seed}")
    zero_share = float(chosen["zero_day_share"].mean()) \
        if "zero_day_share" in chosen else float("nan")
    if reg_hi > 1.0:
        print(f"  regularity drawn per profile in [{reg_lo:g}, {reg_hi:g}] "
              f"({reg_source})")
    else:
        print(f"  regularity off ({reg_source})")
    if np.isfinite(zero_share):
        print(f"  this typology draws nothing on {100 * zero_share:.0f}% of its "
              f"metered days")
    if identity > 0.0:
        print(f"  {100 * identity ** 2:.0f}% of the within-bin variance is held "
              f"by the point rather than by the hour")
    print(f"  strata used: {', '.join(chosen['key'])}")
    thin = sorted({int(m) for s in chosen["thin_months"].dropna().astype(str)
                   for m in s.split(",") if m})
    if thin:
        print(f"  ! months {thin} rest on the backoff rather than on their own "
              f"observations for this typology")

    #Lorenzo Giannuzzo: profiles are split across the strata in proportion to the real population
    weights = chosen["n_pods"].to_numpy(dtype="float64")
    weights = weights / weights.sum()
    counts = rng.multinomial(n_profiles, weights)

    rows, i = [], 0
    for (_, meta), n_here in zip(chosen.iterrows(), counts):
        if n_here == 0:
            continue
        model = load_model(models, meta["key"])
        #Lorenzo Giannuzzo: the anchor and the individual table have to come from the same real
        # point, otherwise a shop's habits are pinned onto another one's size
        anchors = model["annual_kWh"]
        powers = model["power_kW"]
        for _ in range(int(n_here)):
            j = int(rng.integers(0, len(anchors)))
            shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0
            regularity = (float(rng.uniform(reg_lo, reg_hi))
                          if reg_hi > reg_lo else reg_lo)
            anchor_pod = str(model["pods"][j]) if "pods" in model else ""
            pc = (pod_counts_for(model, anchor_pod, days_all, curves_all)
                  if personal and anchor_pod else None)
            closure = 0
            if "closure_class" in model and j < len(model["closure_class"]):
                closure = int(model["closure_class"][j])
            values = walk(model, cal, rng, shift, pc, mix_k, regularity,
                          identity, closure)
            #Lorenzo Giannuzzo: The year is placed at the size of its anchor, so the generated
            # point carries the annual energy of the real point it was drawn
            # from. Dividing by the mean of the stratum instead only gets there
            # in expectation over many draws: the walk has a heavy tailed total
            # of its own, the anchors have another, and the product of the two
            # spreads the generated population far wider than the real one. In
            # the run that produced this comment the anchors spanned a factor of
            # 123,000 and the profiles they produced a factor of 2,020,000, one
            # of them holding 58 percent of all the energy generated and sitting
            # clipped at its contractual limit for half the year.
            total = float(values.sum())
            anchor = float(anchors[j])
            factor = (anchor / total) if total > 0 else 1.0
            values = values * factor
            #Lorenzo Giannuzzo: The bins and their emission are pooled over the stratum, so a small
            # point can draw the level of a large one and come out above a power
            # it could never physically reach. The metered curves are censored at
            # the same limit in stage 1, so applying it here is not a correction
            # of the model but the same physical constraint, and how much of the
            # year it touches is reported rather than hidden.
            p_anchor = float(powers[j])
            clipped = 0.0
            if np.isfinite(p_anchor) and p_anchor > 0:
                limit = p_anchor * power_margin / 4.0
                target = float(values.sum())
                clipped = float((values > limit).mean())
                #Lorenzo Giannuzzo: clip, then give the removed energy back to the quarters that
                # still have room, then clip again: two passes bring the annual
                # total back to the anchor without letting any quarter through
                for _ in range(2):
                    values = np.minimum(values, limit)
                    room = values < limit
                    short = target - float(values.sum())
                    head = float(values[room].sum())
                    if short <= 0 or head <= 0:
                        break
                    values[room] *= 1.0 + short / head
                values = np.minimum(values, limit)

            frame = (pd.DataFrame({"timestamp": cal["timestamp"], "kWh": values})
                     if resolution == "15min" else to_hourly(values, cal))
            frame["kWh"] = frame["kWh"].round(6)
            name = f"{key_typ}_{i:03d}.csv"
            frame.to_csv(out / name, index=False)
            rows.append({
                "file": name, "typology": key_typ, "level": ateco_level,
                "model_key": meta["key"], "resolution": resolution, "year": year,
                "source_pod": anchor_pod,
                "seed": seed, "anchor_power_kW": float(powers[j]),
                "anchor_annual_kWh": round(anchor, 1),
                "clock_shift_quarters": shift,
                "regularity": round(regularity, 3),
                "level_identity": round(identity, 3),
                "closure_class": closure,
                "anchor_pod": anchor_pod,
                "personal_chain": bool(pc is not None),
                "generated_annual_kWh": round(float(frame["kWh"].sum()), 1),
                "unscaled_annual_kWh": round(float(total), 1),
                "peak_kW": round(float(frame["kWh"].max())
                                 * (4 if resolution == "15min" else 1), 3),
                "clipped_share": round(clipped, 4),
                "zero_share": round(float((frame["kWh"] <= 0).mean()), 4),
            })
            i += 1

    man = pd.DataFrame(rows)
    man.to_csv(out / "generation_manifest.csv", index=False)
    print(f"\n  {len(man)} profiles written to {out}")
    print(f"  annual energy: median {man['generated_annual_kWh'].median():,.0f} kWh, "
          f"range {man['generated_annual_kWh'].min():,.0f} to "
          f"{man['generated_annual_kWh'].max():,.0f}")

    if validation:
        from .validation import validate
        validate(man, chosen, out, cfg, year)
    print()
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic load profiles.")
    ap.add_argument("--typology", required=True)
    ap.add_argument("--ateco-level", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--size-range", type=float, nargs=2, default=None,
                    metavar=("MIN_KW", "MAX_KW"))
    ap.add_argument("--n", type=int, default=10, dest="n_profiles")
    ap.add_argument("--resolution", default="15min", choices=("15min", "1h"))
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--outdir", default="results/synthetic")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation", action="store_true")
    a = ap.parse_args()
    generate(a.typology, a.ateco_level,
             tuple(a.size_range) if a.size_range else None,
             a.n_profiles, a.resolution, a.year, a.outdir, a.seed, a.validation)


if __name__ == "__main__":
    main()