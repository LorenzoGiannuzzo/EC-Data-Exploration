"""Comparison stage, Section 2.5.

Two blocks, both on the hourly grid of a reference calendar year.

  B1  every data-driven profile against every admissible national profile. Answers how
      close the published profiles are to the shapes the metered data actually produce.

  B2  every point of delivery against the national profile the regulation assigns to it,
      and against its own data-driven profile as the counterfactual. Answers what the
      published profiles cost in misallocated energy, which is the audit proper and the
      quantity Eq. 15 and Eq. 16 translate into physical and monetary terms.

  B3  the same residual taken on the aggregate rather than on the point. Settlement does
      not allocate one user at a time, and errors that are large per user cancel across
      users when they are independent, so a family can be indistinguishable from another
      at the point and clearly worse at the perimeter. Two scopes are reported: the
      portfolio of a single published class, which rewards a fine partition, and the whole
      perimeter of a family, where the granularity of the partition drops out and only the
      shapes are compared. Each is read twice, at the hourly resolution the imbalance is
      settled on and on the average day of the month.

Both blocks are run under two information settings, because the national profiles are not
given the same inputs by the regulation:

  S0  annual. Each profile receives the annual energy of the user and spreads it over the
      year on its own. The GSE profiles cannot enter: each of their months sums to one
      independently, so they carry no information on how the year splits across months and
      are undefined at this setting. That absence is a result, not a gap.

  S1  monthly. Each profile receives the true monthly energy of the user and spreads it
      over that month. All three objects can play, and the comparison is the one the
      settlement procedure actually performs.

Metrics are computed on hourly values throughout. The settings differ in the scale each
curve is given before the metrics are taken, never in the resolution.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import assignment, calendar as C, national  # noqa: E402
from common.cache import check_manifest  # noqa: E402
from common.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
DATA = ROOT.parent / "data"
OUT = ROOT / "paper_results" / "comparison_results"

PROVINCE = "Trento"
REFERENCE_YEAR = 2025          # the year the GSE workbook refers to
GSE_TREATMENT = "monorario"    # or "fasce"
# Only the two categories that describe the population under study. Public lighting and
# vehicle charging are 84 points out of 7,524 and their four-letter codes cannot be told
# apart with confidence, so they are left out of the comparison rather than guessed at.
GSE_PROFILES = ("PDMM", "PAUM", "PDMF", "PAUF")
# A profile is residential or not, which fixes which data-driven profiles it may be
# compared against in the figures.
GSE_IS_RESIDENTIAL = {"PDMM": True, "PDMF": True, "PAUM": False, "PAUF": False}
# Share of points that must be domestic for a data-driven profile to count as residential.
RESIDENTIAL_THRESHOLD = 0.80
USE_M_FAMILY = False           # sensitivity only, see common/assignment.py
IMBALANCE_PRICE = 15.0         # EUR/MWh, Eq. 16. Placeholder, set from the market data.
MIN_HOURS_PER_MONTH = 240      # a user-month below this is not compared
MAKE_FIGURES = True            # draw the figures at the end of the stage
# B3 is accumulated twice: once per reference, and once over the whole perimeter a
# family is asked to settle. The second scope carries this label.
SYSTEM_LABEL = "__all__"


# ----------------------------------------------------------------------------- metrics
def metrics(obs: np.ndarray, ref: np.ndarray, hours_of_day: np.ndarray) -> dict:
    """Five measures, chosen so that no one of them is implied by the others.

    obs and ref are hourly energies over the same hours, in the same unit. Shares are
    taken internally, so any common scaling of both leaves every measure unchanged except
    nRMSE, which is normalised by the mean of obs.
    """
    o, r = np.asarray(obs, float), np.asarray(ref, float)
    so, sr = o.sum(), r.sum()
    if so <= 0 or sr <= 0:
        return {k: np.nan for k in
                ("nrmse", "total_variation", "peak_hour_shift", "par_ratio", "pearson")}
    po, pr = o / so, r / sr
    rmse = float(np.sqrt(np.mean((o - r) ** 2)))
    tv = float(0.5 * np.abs(po - pr).sum())
    prof_o = np.bincount(hours_of_day, weights=o, minlength=24)
    prof_r = np.bincount(hours_of_day, weights=r, minlength=24)
    shift = int(prof_o.argmax()) - int(prof_r.argmax())
    shift = (shift + 12) % 24 - 12                      # signed, shortest way round
    par_o = o.max() / o.mean()
    par_r = r.max() / r.mean()
    if o.std() == 0 or r.std() == 0:
        rho = np.nan
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = float(np.corrcoef(po, pr)[0, 1])
    return {"nrmse": rmse / (so / len(o)),
            "total_variation": tv,
            "peak_hour_shift": shift,
            "par_ratio": float(par_r / par_o),
            "pearson": rho}


# ------------------------------------------------------------------- data driven curves
def load_ddslp(cache: Path) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (profiles[group, cell, hour], cell order, group ids).

    profiles.npy holds quarter-hourly shapes summing to one within each cell. The cell
    order along axis 1 is not stored with the array, so it is recovered from the order the
    weights table was written in and checked for consistency across groups.
    """
    arr = np.load(cache / "profiles.npy")
    w = pd.read_parquet(cache / "profile_weights.parquet")
    groups = sorted(w["group"].unique())
    orders = {g: list(w.loc[w["group"] == g, "cell"]) for g in groups}
    first = orders[groups[0]]
    if any(orders[g] != first for g in groups):
        raise ValueError("cell order differs across groups in profile_weights.parquet")
    if arr.shape[:2] != (len(groups), len(first)):
        raise ValueError(f"profiles.npy is {arr.shape}, expected "
                         f"({len(groups)}, {len(first)}, 96)")
    hourly = arr.reshape(arr.shape[0], arr.shape[1], 24, 4).sum(axis=3)
    return hourly, first, [int(g) for g in groups]


def hcal_cell_key(hcal: pd.DataFrame, cells: list[str]) -> pd.Series:
    """The cell label of every hour of the reference calendar.

    generation.py writes a cell as "<period>|<daytype>", where the period is a season or a
    month depending on generation.grid. The calendar already carries the seasonal label;
    the monthly one is rebuilt here from the month and the day type, so that a profile
    built on either grid can be expanded without a second calendar and the two grids can
    be compared on the same footing.
    """
    def is_month(c: str) -> bool:
        p = c.split("|")[0]
        return len(p) == 3 and p[0] == "M" and p[1:].isdigit()

    if not cells or not all(is_month(c) for c in cells):
        return hcal["cell"].astype(str)
    return (hcal["month"].astype(int).map(lambda m: f"M{m:02d}")
            + "|" + hcal["daytype"].astype(str))


def ddslp_hourly_year(cache: Path, hcal: pd.DataFrame) -> pd.DataFrame:
    """Expand each data-driven profile onto the reference calendar.

    The weight of a cell is the share of annual energy the profile places in that cell. It
    is spread evenly across the calendar days belonging to the cell, so the resulting
    hourly vector sums to one over the year by construction.
    """
    hourly, cells, groups = load_ddslp(cache)
    w = pd.read_parquet(cache / "profile_weights.parquet")
    wmap = {(int(r.group), r.cell): float(r.weight) for r in w.itertuples()}
    key = hcal_cell_key(hcal, cells)
    days_per_cell = (hcal.assign(_cell=key).drop_duplicates("date")["_cell"]
                     .value_counts().to_dict())
    cell_idx = {c: i for i, c in enumerate(cells)}
    unmatched = [c for c in cells if days_per_cell.get(c, 0) == 0]
    if unmatched:
        print(f"  ! {len(unmatched)} data-driven cells find no day on the reference "
              f"calendar ({', '.join(unmatched[:4])}...): their weight is dropped")
    out = {}
    for gi, g in enumerate(groups):
        v = np.empty(len(hcal))
        for cell, idx in cell_idx.items():
            m = (key == cell).to_numpy()
            n = days_per_cell.get(cell, 0)
            if n == 0:
                v[m] = 0.0
                continue
            shape = hourly[gi, idx]
            v[m] = np.tile(shape, n) * wmap[(g, cell)] / n
        out[f"DDSLP_{g}"] = v
    return pd.DataFrame(out)


# --------------------------------------------------------------- national curves, hourly
def arera_hourly_year(tab: pd.DataFrame, hcal: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """(power class, residency) -> hourly kWh per point over the reference calendar."""
    key = tab.set_index(["power_class", "residency", "month", "daytype", "hour"])["kWh"].sort_index()
    curves = {}
    for (cls, res), _ in tab.groupby(["power_class", "residency"], observed=True):
        try:
            sub = key.loc[(cls, res)]
        except KeyError:
            continue
        v = sub.reindex(pd.MultiIndex.from_arrays(
            [hcal["month"], hcal["daytype"], hcal["hour"]])).to_numpy(dtype=float)
        if np.isnan(v).any():
            continue
        curves[(cls, res)] = v
    return curves


def gse_hourly_year(gse: pd.DataFrame, hcal: pd.DataFrame) -> dict[str, np.ndarray]:
    """Profile name -> hourly share, aligned to the reference calendar.

    The GSE workbook is a calendar year in its own right. Alignment is on (month, day,
    hour), which is exact when the reference year is the year of the workbook.
    """
    cols = [c for c in gse.columns
            if c not in ("year", "month", "day", "hour") and c in GSE_PROFILES]
    # The workbook is a civil calendar and therefore carries a 23-hour and a 25-hour day.
    # They are dropped here for the same reason preprocessing drops them (Section 2.2):
    # they are neither missing values nor duplicates, and any realignment convention would
    # need its own justification.
    n_hours = gse.groupby(["month", "day"])["hour"].transform("size")
    gse = gse[n_hours == 24]
    g = gse.set_index(["month", "day", "hour"])
    idx = pd.MultiIndex.from_arrays([hcal["month"], hcal["date"].dt.day, hcal["hour"]])
    out = {}
    for c in cols:
        v = g[c].reindex(idx).to_numpy(dtype=float)
        if np.isnan(v).any():
            n = int(np.isnan(v).sum())
            print(f"  ! GSE {c}: {n} hours unmatched on the reference calendar, skipped")
            continue
        out[c] = v
    return out


# ---------------------------------------------------------------------- observed curves
def observed_hourly(days: pd.DataFrame, dictionary: np.ndarray, cache: Path,
                    hcal: pd.DataFrame, year: int,
                    user_vectors: pd.DataFrame | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Reconstruct each point's hourly curve over the reference year.

    Three sources, in order of fidelity, and the one used is reported:

      1. shapes.npy. Preprocessing writes the normalised curve of every admitted
         day, and days.parquet carries the row of each day into it. Multiplying
         that curve by the day's energy returns the measured day itself, with no
         quantisation. This is what the audit should be run against and it is
         what is used whenever the file is present.

      2. day_codeword.npy, when it covers every admitted day. Each day is then
         represented by its codeword rather than by its own curve. Note that when
         the first clustering stage runs on POD-months this array has one entry
         per POD-month, not per day, and it cannot serve here: a whole month would
         collapse onto a single shape and every day of that month would differ
         from the measurement in the same direction.

      3. The user's own mixture of codewords, which keeps the average shape and
         removes all variation within the year. It flatters every profile equally
         and is a last resort.
    """
    d = days[(days["date"].dt.year == year) & days["has_shape"]].copy()

    cal_days = hcal.drop_duplicates("date")["date"].reset_index(drop=True)
    slot_of = {(t.month, t.day): i for i, t in enumerate(cal_days)}
    d["slot"] = [slot_of.get((m, dd), -1) for m, dd in
                 zip(d["date"].dt.month, d["date"].dt.day)]
    d = d[d["slot"] >= 0]

    pods = np.sort(d["pod"].unique())
    pod_ix = {p: i for i, p in enumerate(pods)}
    n_days = len(cal_days)
    mat = np.zeros((len(pods), n_days, 24), dtype=np.float32)
    seen = np.zeros((len(pods), n_days), dtype=bool)
    pi = d["pod"].map(pod_ix).to_numpy()
    si = d["slot"].to_numpy()
    energy = d["energy"].to_numpy(dtype=np.float32)

    shp_path = cache / "shapes.npy"
    n_admitted = int(days["has_shape"].sum())
    if shp_path.exists():
        arr = np.load(shp_path, mmap_mode="r")
        if arr.shape[0] != n_admitted:
            raise ValueError(f"shapes.npy has {arr.shape[0]} rows, {n_admitted} "
                             "admitted days expected; the cache is inconsistent")
        mode = "measured day shapes (shapes.npy)"
        idx = d["shape_idx"].to_numpy()
        order = np.argsort(idx)              # memmap reads are sequential
        for lo in range(0, len(order), 200_000):
            sel = order[lo:lo + 200_000]
            block = np.asarray(arr[idx[sel]], dtype=np.float32)
            block = block.reshape(len(sel), 24, 4).sum(axis=2)
            mat[pi[sel], si[sel], :] = block * energy[sel, None]
            seen[pi[sel], si[sel]] = True
    else:
        shp = dictionary.reshape(dictionary.shape[0], 24, 4).sum(axis=2)
        cw = None
        cw_path = cache / "day_codeword.npy"
        if cw_path.exists():
            a = np.load(cw_path)
            if a.shape[0] == n_admitted:
                cw = a
            else:
                print(f"  ! day_codeword.npy holds {a.shape[0]} entries against "
                      f"{n_admitted} admitted days: it indexes another unit "
                      "(POD-months, most likely), not usable here")
        if cw is not None:
            mode = "per-day codewords"
            day_shape = shp[cw[d["shape_idx"].to_numpy()]]
        else:
            if user_vectors is None:
                raise ValueError("no shapes.npy, no per-day codewords, no fallback")
            mode = "per-user codeword mixture (no measured shapes available)"
            fcols = [c for c in user_vectors.columns if c.startswith("f_")]
            f = user_vectors.set_index("pod")[fcols].reindex(pods).fillna(0.0).to_numpy()
            f = np.where(f.sum(1, keepdims=True) > 0, f, 1.0 / len(fcols))
            f = f / f.sum(1, keepdims=True)
            day_shape = (f @ shp)[pi]
        mat[pi, si, :] = day_shape * energy[:, None]
        seen[pi, si] = True

    return pods, mat.reshape(len(pods), -1), seen, mode


# ------------------------------------------------------------- scaling a reference curve
def is_banded(name: str) -> bool:
    """A GSE profile published per tariff band rather than at a single rate."""
    return name.startswith("GSE") and name.rstrip().endswith("F")


def scale_reference(ref: np.ndarray, obs: np.ndarray, bands: np.ndarray,
                    banded: bool) -> np.ndarray:
    """Give a reference profile the energy the regulation would give it.

    A single-rate profile receives one number for the period, the metered energy, and
    spreads it over its own shape. A banded profile receives three, one per tariff band,
    because its published values are normalised to one *within each band* and are
    meaningless until each band is given its own quantity. Handing a banded profile a
    single total would misread its normalisation and flatter it or penalise it depending
    on the user, which is why the two are scaled differently here rather than uniformly.
    """
    out = np.zeros_like(ref, dtype=float)
    if not banded:
        tot = ref.sum()
        return ref * (obs.sum() / tot) if tot > 0 else out
    for b in (1, 2, 3):
        m = bands == b
        if not m.any():
            continue
        rs = ref[m].sum()
        if rs > 0:
            out[m] = ref[m] * (obs[m].sum() / rs)
    return out


def ddslp_kind(users: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Composition of each data-driven profile, and whether it counts as residential.

    The clustering pools every point, so a profile is not residential or non-residential
    by construction: it becomes one or the other through who ends up in it. The share is
    reported by number of points and by energy, because the two diverge sharply here and
    the divergence is itself worth stating.
    """
    from common.assignment import gse_category
    u = users.assign(kind=gse_category(users).values)
    m = groups.merge(u[["pod", "kind", "E"]], on="pod", how="inner")
    rows = []
    for g, x in m.groupby("group"):
        dom = x["kind"] == "domestic"
        rows.append({"group": int(g), "profile": f"DDSLP_{int(g)}", "n_pod": len(x),
                     "share_domestic_pod": float(dom.mean()),
                     "share_domestic_energy": float(x.loc[dom, "E"].sum() / x["E"].sum()),
                     "mean_annual_kWh": float(x["E"].mean())})
    out = pd.DataFrame(rows)
    out["residential"] = out["share_domestic_pod"] >= RESIDENTIAL_THRESHOLD
    return out


# ------------------------------------------------------------------------------ blocks
def block_b1(dd: pd.DataFrame, arera: dict, gse: dict, hcal: pd.DataFrame) -> pd.DataFrame:
    """Every data-driven profile against every national profile, annual and monthly."""
    hod = hcal["hour"].to_numpy()
    month = hcal["month"].to_numpy()
    bands = hcal["band"].to_numpy()
    rows = []
    for name, v in dd.items():
        v = v.to_numpy()
        for (cls, res), a in arera.items():
            rows.append({"ddslp": name, "national": f"ARERA {cls} {res}",
                         "source": "ARERA", "setting": "S0_annual",
                         **metrics(v, scale_reference(a, v, bands, False), hod)})
        for setting_ref, label in ((arera, "ARERA"), (gse, "GSE")):
            for k, r in setting_ref.items():
                nm = f"ARERA {k[0]} {k[1]}" if label == "ARERA" else f"GSE {k}"
                banded = is_banded(nm)
                per_month, wts = [], []
                for m in range(1, 13):
                    sel = month == m
                    ref = scale_reference(r[sel], v[sel], bands[sel], banded)
                    mm = metrics(v[sel], ref, hod[sel])
                    per_month.append(mm)
                    wts.append(v[sel].sum())
                wts = np.asarray(wts, dtype=float)
                agg = {}
                for key in per_month[0]:
                    vals = np.array([p[key] for p in per_month], dtype=float)
                    ok = np.isfinite(vals) & (wts > 0)
                    # a month whose metric is undefined (a flat reference has no
                    # correlation to speak of) is dropped from the average, not counted
                    # as a zero, which would silently reward the flat profile
                    agg[key] = float(np.average(vals[ok], weights=wts[ok])) if ok.any() else np.nan
                agg["peak_hour_shift"] = float(np.median(
                    [p["peak_hour_shift"] for p in per_month]))
                rows.append({"ddslp": name, "national": nm, "source": label,
                             "setting": "S1_monthly", **agg})
    return pd.DataFrame(rows)


def block_b2(pods: np.ndarray, obs: np.ndarray, seen: np.ndarray,
             assign: pd.DataFrame, groups: pd.DataFrame,
             dd: pd.DataFrame, arera: dict, gse: dict,
             hcal: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Each point against the profile the regulation gives it, and against its own.

    Run at S1: every profile receives the true monthly energy of the point, so the residual
    is what the shape alone fails to explain.
    """
    hod = hcal["hour"].to_numpy()
    month = hcal["month"].to_numpy()
    bands = hcal["band"].to_numpy()
    n_year = len(hod)
    a = assign.set_index("pod")
    g = groups.set_index("pod")["group"]
    E = users.set_index("pod")["E"]
    rows = []
    # Settlement does not allocate one user at a time: it allocates a portfolio and
    # settles the residual of the whole. Errors that are large per user cancel across
    # users when they are independent, so the aggregate residual is the quantity the
    # imbalance is actually computed on, and it is accumulated here alongside the
    # per-user figures rather than in a second pass over the data.
    agg: dict[tuple[str, str, int], list] = {}
    for i, pod in enumerate(pods):
        if pod not in a.index:
            continue
        rec = a.loc[pod]
        o_all = obs[i]
        refs = {}
        col = rec["gse_column"]
        if col in gse:
            refs["GSE"] = (gse[col], col, bool(rec["gse_uncertain"]))
        if rec["arera_applicable"]:
            k = (rec["arera_class"], rec["arera_residency"])
            if k in arera:
                refs["ARERA"] = (arera[k], f"{k[0]} {k[1]}", False)
        if pod in g.index:
            nm = f"DDSLP_{int(g.loc[pod])}"
            if nm in dd:
                refs["DDSLP"] = (dd[nm].to_numpy(), nm, False)
        if not refs:
            continue
        for m in range(1, 13):
            sel = (month == m)
            hours = sel & np.repeat(seen[i], 24)
            if hours.sum() < MIN_HOURS_PER_MONTH:
                continue
            o = o_all[hours]
            if o.sum() <= 0:
                continue
            # A portfolio comparison is only fair on the user-months every family could
            # be evaluated on: ARERA covers domestic points alone, and a family credited
            # with the easy months only would win on the sample, not on the method.
            usable = {src for src, (ref, _, _) in refs.items() if ref[hours].sum() > 0}
            complete = len(usable) == len(refs) and len(refs) == 3
            for src, (ref, label, unc) in refs.items():
                r = ref[hours]
                if r.sum() <= 0:
                    continue
                r = scale_reference(r, o, bands[hours], is_banded(f"GSE {label}")
                                    if src == "GSE" else False)
                if r.sum() <= 0:
                    continue
                mm = metrics(o, r, hod[hours])
                if not complete:
                    agg_ok = False
                else:
                    agg_ok = True
                if not agg_ok:
                    rows.append({"pod": pod, "month": m, "source": src,
                                 "reference": label, "uncertain_assignment": unc,
                                 "month_energy_kWh": float(o.sum()), **mm})
                    continue
                # accumulated at full hourly resolution rather than folded onto an
                # average day: settlement is hourly, and pooling the hours of a month
                # cancels exactly the day-to-day part of the residual it charges for.
                # The fold is recovered afterwards, so both readings are reported.
                idxh = np.flatnonzero(hours)
                for lbl in (label, SYSTEM_LABEL):
                    k2 = (src, lbl, m)
                    if k2 not in agg:
                        agg[k2] = [np.zeros(n_year), np.zeros(n_year), 0.0, 0]
                    acc = agg[k2]
                    np.add.at(acc[0], idxh, o)
                    np.add.at(acc[1], idxh, r)
                    acc[2] += float(o.sum())
                    acc[3] += 1
                rows.append({"pod": pod, "month": m, "source": src, "reference": label,
                             "uncertain_assignment": unc,
                             "month_energy_kWh": float(o.sum()), **mm})
    out = pd.DataFrame(rows)
    if len(out):
        out["misallocated_kWh"] = out["total_variation"] * out["month_energy_kWh"]
        out["imbalance_cost_EUR"] = out["misallocated_kWh"] / 1000.0 * IMBALANCE_PRICE

    arows = []
    for (src, label, m), (so, sr, tot, n) in agg.items():
        if so.sum() <= 0 or sr.sum() <= 0:
            continue
        po, pr = so / so.sum(), sr / sr.sum()
        tv = float(0.5 * np.abs(po - pr).sum())
        # the same residual read on the average day of the month. Total variation cannot
        # increase when hours are pooled, so tv >= tv_mean_day always, and the gap is the
        # share of the misallocation that only the hourly resolution exposes. It is a
        # lower bound on the day-to-day component, not an orthogonal decomposition.
        do = np.bincount(hod, weights=so, minlength=24)
        dr = np.bincount(hod, weights=sr, minlength=24)
        tv_d = float(0.5 * np.abs(do / do.sum() - dr / dr.sum()).sum())
        arows.append({"source": src, "reference": label, "month": m,
                      "scope": "system" if label == SYSTEM_LABEL else "reference",
                      "n_pod": n, "portfolio_energy_kWh": tot,
                      "aggregate_total_variation": tv,
                      "tv_mean_day": tv_d,
                      "tv_day_to_day": tv - tv_d,
                      "aggregate_misallocated_kWh": tv * tot,
                      "aggregate_imbalance_cost_EUR": tv * tot / 1000.0 * IMBALANCE_PRICE})
    return out, pd.DataFrame(arows)


# --------------------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*78}\n  COMPARISON, Section 2.5\n{'='*78}")

    # the profiles audited here have to come from the configuration now in
    # force: an audit reporting a dictionary that no longer exists fails silently
    man = check_manifest(CACHE, load_config()["clustering"], "comparison")

    days = pd.read_parquet(CACHE / "days.parquet")
    users = pd.read_parquet(CACHE / "users.parquet")
    groups = pd.read_parquet(CACHE / "groups.parquet")
    dictionary = np.load(CACHE / "dictionary.npy")
    print(f"  dictionary: {dictionary.shape[0]} codewords "
          f"(cache written with unit {man.get('shape_unit')}, "
          f"K = {man.get('n_profiles')})")

    smap = C.season_map_from_days(days)
    conv = C.check_daytype_convention(days)
    print(f"  holidays folded into sunday: {conv['holidays_folded_into_sunday']} "
          f"({conv['labelled_sunday']}/{conv['weekday_holidays_observed']})")
    cal = C.build_calendar(REFERENCE_YEAR, smap)
    hcal = C.hourly_index(cal)

    print("  loading national profiles")
    tab, prov = national.load_arera(DATA, PROVINCE, cache_dir=CACHE)
    gse_raw = national.load_gse(DATA, cache_dir=CACHE)
    prov.to_csv(OUT / "arera_provenance.csv", index=False)
    national.gse_normalisation_report(gse_raw).to_csv(
        OUT / "gse_normalisation.csv", index=False)

    dd = ddslp_hourly_year(CACHE, hcal)
    arera = arera_hourly_year(tab, hcal)
    gse = gse_hourly_year(gse_raw, hcal)
    print(f"  {dd.shape[1]} data-driven, {len(arera)} ARERA, {len(gse)} GSE curves")

    assign = assignment.build(users, treatment=GSE_TREATMENT, use_m_family=USE_M_FAMILY)
    assign.to_csv(OUT / "assignment.csv", index=False)

    kinds = ddslp_kind(users, groups)
    kinds.to_csv(OUT / "ddslp_composition.csv", index=False)
    print("\n" + kinds.round(3).to_string(index=False) + "\n")

    b1 = block_b1(dd, arera, gse, hcal)
    b1.to_csv(OUT / "b1_ddslp_vs_national.csv", index=False)
    print(f"  B1: {len(b1)} pairs")

    uv = pd.read_parquet(CACHE / "user_vectors.parquet")
    # The B2 comparison is only as good as the curve it calls observed; the mode is
    # printed and written to the summary so a result can never be read without it.
    pods, obs, seen, mode = observed_hourly(days, dictionary, CACHE, hcal,
                                            REFERENCE_YEAR, user_vectors=uv)
    print(f"  reconstructed {len(pods)} points over {REFERENCE_YEAR} [{mode}]")
    b2, b3 = block_b2(pods, obs, seen, assign, groups, dd, arera, gse, hcal, users)
    b2.to_csv(OUT / "b2_pod_month.csv", index=False)
    b3.to_csv(OUT / "b3_portfolio_month.csv", index=False)
    if len(b3):
        # two scopes. `reference` is the portfolio of a single published class, and it
        # rewards a fine partition; `system` pools every class of a family into the one
        # perimeter the imbalance is settled on, where the granularity of the partition
        # no longer enters the residual and only the shapes are left to compare.
        for scope, note in (("reference", "the portfolio each profile is asked to settle"),
                            ("system", "the whole perimeter of each family")):
            x3 = b3[b3["scope"] == scope]
            if not len(x3):
                continue
            pf = (x3.groupby("source")
                    .apply(lambda x: pd.Series({
                        "n_portfolio_months": len(x),
                        "tv_energy_weighted": float(np.average(
                            x["aggregate_total_variation"],
                            weights=x["portfolio_energy_kWh"])),
                        "tv_mean_day": float(np.average(
                            x["tv_mean_day"], weights=x["portfolio_energy_kWh"])),
                        "misallocated_kWh": float(x["aggregate_misallocated_kWh"].sum()),
                        "imbalance_cost_EUR": float(x["aggregate_imbalance_cost_EUR"].sum()),
                    }), include_groups=False))
            pf.to_csv(OUT / f"b3_summary_{scope}.csv")
            print(f"\n  B3 [{scope}], {note}:")
            print(pf.round(4).to_string())

    if len(b2):
        # Totals are only comparable on the user-months every source could be evaluated
        # on: ARERA covers domestic points alone, so the raw sums span different samples.
        wide = b2.pivot_table(index=["pod", "month"], columns="source",
                              values="total_variation")
        common = wide.dropna().index
        b2["common_set"] = pd.MultiIndex.from_frame(b2[["pod", "month"]]).isin(common)
        b2.to_csv(OUT / "b2_pod_month.csv", index=False)
        cs = (b2[b2["common_set"]].groupby("source")
                .agg(n=("pod", "size"),
                     tv_median=("total_variation", "median"),
                     misallocated_kWh=("misallocated_kWh", "sum"),
                     imbalance_cost_EUR=("imbalance_cost_EUR", "sum")))
        cs.to_csv(OUT / "b2_summary_common_set.csv")
        print("\n  common set of user-months (every source evaluated):")
        print(cs.round(4).to_string())
        s = (b2.groupby("source")
               .agg(n=("pod", "size"),
                    tv_median=("total_variation", "median"),
                    tv_p90=("total_variation", lambda x: x.quantile(0.90)),
                    nrmse_median=("nrmse", "median"),
                    pearson_median=("pearson", "median"),
                    misallocated_kWh=("misallocated_kWh", "sum"),
                    imbalance_cost_EUR=("imbalance_cost_EUR", "sum")))
        s.to_csv(OUT / "b2_summary.csv")
        print("\n" + s.round(4).to_string())

    with open(OUT / "summary.txt", "w", encoding="utf-8") as fh:
        fh.write(f"reference year          {REFERENCE_YEAR}\n")
        fh.write(f"province                {PROVINCE}\n")
        fh.write(f"codewords               {dictionary.shape[0]}\n")
        fh.write(f"season map              {smap}\n")
        fh.write(f"holidays as sunday      {conv}\n")
        fh.write(f"GSE treatment           {GSE_TREATMENT}\n")
        fh.write(f"GSE M family used       {USE_M_FAMILY}\n")
        fh.write(f"imbalance price EUR/MWh {IMBALANCE_PRICE}\n")
        fh.write(f"observed curve mode     {mode}\n\n")
        fh.write(prov.to_string(index=False) + "\n")

    print(f"\n  results in {OUT}   ({time.time()-t0:.0f}s)")

    if MAKE_FIGURES:
        # Drawn here so that running the stage produces the tables and the figures that
        # go with them in one go. A failure while plotting must not discard the results
        # that were just computed, so it is reported and swallowed. The figures remain
        # available as a stage of their own through `main.py --stage figures`.
        try:
            import figures
            figures.main()
        except Exception as exc:
            print(f"\n  ! figures not produced: {type(exc).__name__}: {exc}")
            print("    the comparison tables above are unaffected; "
                  "rerun with  python main.py --stage figures\n")
    print()


if __name__ == "__main__":
    main()