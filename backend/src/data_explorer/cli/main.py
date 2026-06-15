"""`data-explorer` command-line entry point.

Usage examples:
    data-explorer ingest                       # use default /data/raw
    data-explorer ingest --data-dir ./data
    data-explorer ingest-ateco /data/raw/Note-esplicative-ATECO-2025-italiano-inglese.xlsx
    data-explorer db-check
    data-explorer cluster-test --ateco-l1 DO -k 5
"""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import text

from data_explorer.config import settings
from data_explorer.db.ingestion import (
    ARERA_FILE_TO_POWER_CLASS,
    create_aggregated_profiles_view, ingest_arera_directory, ingest_arera_file,
    ingest_ateco_lookup, ingest_data_directory,
    ingest_gse_profiles, migrate_reference_tables,
    refresh_aggregated_profiles_view,
)
from data_explorer.db.session import SessionLocal, engine

app = typer.Typer(
    help="Data Explorer — load, query, and serve electrical-load datasets.",
    no_args_is_help=True,
)
console = Console()


# ── ingest ───────────────────────────────────────────────────────────────────
@app.command()
def ingest(
    data_dir: Path = typer.Option(
        Path(settings.raw_data_dir),
        "--data-dir", "-d",
        help="Root folder with monthly sub-directories (ago24/, set24/, ...).",
    ),
):
    """Bulk-load metadata + measurements from the legacy `data/` folder."""
    console.print(f"[bold]Ingesting from[/bold] {data_dir}")
    stats = ingest_data_directory(data_dir)

    table = Table(title="Ingestion summary", show_header=True, header_style="bold cyan")
    table.add_column("Item",  justify="left")
    table.add_column("Count", justify="right")
    table.add_row("Metadata files",     str(stats["meta_files"]))
    table.add_row("Metadata rows",      f'{stats["meta_rows"]:,}')
    table.add_row("Measurement files",  str(stats["meas_files"]))
    table.add_row("Measurement rows",   f'{stats["meas_rows"]:,}')
    console.print(table)


# ── ingest-ateco ─────────────────────────────────────────────────────────────
@app.command("ingest-ateco")
def ingest_ateco(
    xlsx: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True,
        help="Path to the ATECO 2025 explanatory notes Excel.",
    ),
):
    """Load the official ATECO lookup table (codes → descriptions)."""
    with SessionLocal() as session:
        n = ingest_ateco_lookup(session, xlsx)
    console.print(f"[green]Loaded[/green] {n:,} ATECO entries")


# ── db-check ─────────────────────────────────────────────────────────────────
@app.command("db-check")
def db_check():
    """Show row counts for each main table — useful as a quick health check."""
    with engine.connect() as conn:
        tables = [
            "pod_metadata", "measurements", "ateco_lookup",
            "reference_profiles_gse", "reference_profiles_arera",
        ]
        table = Table(title="Row counts", show_header=True, header_style="bold cyan")
        table.add_column("Table", justify="left")
        table.add_column("Rows",  justify="right")
        for t in tables:
            try:
                n = conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
                table.add_row(t, f"{n:,}")
            except Exception as e:
                table.add_row(t, f"[red]error: {e}[/red]")
    console.print(table)


# ── cluster-test ─────────────────────────────────────────────────────────────
@app.command("cluster-test")
def cluster_test(
    ateco_l1:   str = typer.Option(None, "--ateco-l1", help="Filter PODs by ATECO L1 code (e.g. 'DO', '47')."),
    k:          int = typer.Option(5, "-k", help="Number of clusters."),
    min_months: int = typer.Option(12, help="Minimum months of data per POD."),
    method:     str = typer.Option("ward", help="Linkage method: ward, average, complete, single."),
):
    """End-to-end smoke test: pull PODs from Postgres, aggregate profiles in SQL,
    cluster them, and print the result. Validates the whole core+db stack."""
    from data_explorer.core.clustering import (
        cluster_ateco_breakdown, cluster_metrics, cluster_profiles, compute_centroids,
        dominant_ateco_per_cluster,
    )
    from data_explorer.core.profiles import filter_all_zero, normalise_profiles, profile_summary
    from data_explorer.db.queries import (
        fetch_aggregated_profiles, fetch_ateco_descriptions, fetch_pod_metadata,
        fetch_pods_by_ateco, fetch_pods_with_data_coverage,
    )

    with SessionLocal() as session:
        # 1. Resolve POD universe
        console.print("[bold]Step 1[/bold] — selecting PODs…")
        coverage_pods = fetch_pods_with_data_coverage(session, min_months=min_months)
        console.print(f"  PODs with ≥{min_months} months of data: {len(coverage_pods):,}")

        if ateco_l1:
            ateco_pods = fetch_pods_by_ateco(session, [ateco_l1], level=1)
            console.print(f"  PODs in ATECO L1 = {ateco_l1!r}: {len(ateco_pods):,}")
            pods = coverage_pods & ateco_pods
        else:
            pods = coverage_pods
        console.print(f"  [green]Final POD set:[/green] {len(pods):,}")

        if len(pods) < k:
            console.print(f"[red]Not enough PODs ({len(pods)}) to form {k} clusters.[/red]")
            raise typer.Exit(code=1)

        # 2. Aggregate profiles IN POSTGRES (the key optimisation)
        console.print("\n[bold]Step 2[/bold] — aggregating daily profiles in PostgreSQL…")
        profiles = fetch_aggregated_profiles(session, pod_ids=pods)
        console.print(f"  Profiles fetched: {len(profiles):,} POD × 96 Q-cols")
        console.print(f"  {profile_summary(profiles)}")

        # 3. Filter + normalise
        console.print("\n[bold]Step 3[/bold] — filter all-zero + normalise…")
        profiles = filter_all_zero(profiles)
        normalised = normalise_profiles(profiles, method="max")
        console.print(f"  After cleaning: {len(normalised):,} usable profiles")

        # 4. Cluster
        console.print(f"\n[bold]Step 4[/bold] — hierarchical clustering (k={k}, method={method})…")
        clusters = cluster_profiles(normalised, n_clusters=k, method=method)

        # 5. Metrics + centroids
        metrics   = cluster_metrics(normalised, clusters)
        centroids = compute_centroids(normalised, clusters)

        t = Table(title="Cluster metrics", show_header=True, header_style="bold cyan")
        t.add_column("Metric"); t.add_column("Value", justify="right")
        t.add_row("Total PODs",     f"{metrics['n_pods']:,}")
        t.add_row("Clusters",       f"{metrics['n_clusters']}")
        t.add_row("Silhouette",
                  f"{metrics['silhouette']:.4f}" if metrics['silhouette'] is not None else "N/A")
        console.print(t)

        sz = Table(title="Cluster sizes", show_header=True, header_style="bold cyan")
        sz.add_column("Cluster", justify="right"); sz.add_column("N. PODs", justify="right")
        for cl, n in metrics["sizes"].items():
            sz.add_row(f"{cl}", f"{n:,}")
        console.print(sz)

        # 6. ATECO breakdown
        console.print("\n[bold]Step 5[/bold] — dominant ATECO per cluster…")
        meta = fetch_pod_metadata(session, pod_ids=list(clusters.index))
        breakdown = cluster_ateco_breakdown(clusters, meta, ateco_level=1)
        top = dominant_ateco_per_cluster(breakdown, top_n=3)
        descriptions = fetch_ateco_descriptions(session)
        top["description"] = top["ateco"].map(
            lambda c: descriptions.get(c, "")[:60] if c else ""
        )

        bd = Table(title="Top-3 ATECO per cluster", show_header=True, header_style="bold cyan")
        bd.add_column("Cluster",      justify="right")
        bd.add_column("ATECO")
        bd.add_column("N. PODs",      justify="right")
        bd.add_column("% of cluster", justify="right")
        bd.add_column("Description")
        for _, r in top.iterrows():
            bd.add_row(str(int(r["cluster"])), str(r["ateco"]),
                       f"{int(r['n_pods']):,}", f"{r['pct_of_cluster']:.1f}%",
                       str(r["description"]))
        console.print(bd)

        console.print("\n[green]✓ Smoke test complete — clustering pipeline works end-to-end.[/green]")


# ── migrate-refs ─────────────────────────────────────────────────────────────
@app.command("migrate-refs")
def migrate_refs():
    """Drop & re-create reference_profiles_* tables with the latest schema.

    Safe because these tables hold re-ingestable data only. Run this before
    `ingest-gse` / `ingest-arera` the first time after pulling Phase 2b.
    """
    with SessionLocal() as session:
        migrate_reference_tables(session)
    console.print("[green]Reference tables re-created with latest schema.[/green]")


# ── refresh-views ────────────────────────────────────────────────────────────
@app.command("refresh-views")
def refresh_views(
    recreate: bool = typer.Option(False, "--recreate",
                                   help="Drop and recreate the view instead of refreshing."),
):
    """Refresh (or recreate) the pod_avg_profile materialized view.

    Run this once after the initial ingestion, and after every new ingest.
    The view pre-aggregates per-POD profiles so clustering is instant.
    """
    with SessionLocal() as session:
        if recreate:
            n = create_aggregated_profiles_view(session)
            console.print(f"[green]Recreated[/green] pod_avg_profile — "
                          f"{n:,} rows")
        else:
            n = refresh_aggregated_profiles_view(session)
            console.print(f"[green]Refreshed[/green] pod_avg_profile — "
                          f"{n:,} rows")


# ── ingest-gse ───────────────────────────────────────────────────────────────
@app.command("ingest-gse")
def ingest_gse(
    xlsx: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, resolve_path=True,
        help="Path to `profili GSE_prelievo_2025.xlsx` (or equivalent).",
    ),
):
    """Load the GSE reference profiles Excel into PostgreSQL."""
    with SessionLocal() as session:
        n = ingest_gse_profiles(session, xlsx)
    console.print(f"[green]Loaded[/green] {n:,} GSE profile rows")


# ── ingest-arera ─────────────────────────────────────────────────────────────
@app.command("ingest-arera")
def ingest_arera(
    data_dir: Path = typer.Option(
        Path(settings.raw_data_dir), "--data-dir", "-d",
        help="Folder containing all five ARERA Excel files.",
    ),
    province: str = typer.Option(
        "Trento", help="Province to import (ARERA files include every province)."
    ),
):
    """Load all five ARERA per-power-class Excels into PostgreSQL."""
    with SessionLocal() as session:
        stats = ingest_arera_directory(session, data_dir, province=province)

    table = Table(title=f"ARERA ingestion ({province})",
                  show_header=True, header_style="bold cyan")
    table.add_column("Power class")
    table.add_column("Rows", justify="right")
    for pc, n in stats.items():
        table.add_row(pc, f"{n:,}")
    console.print(table)


# ── gse-test ─────────────────────────────────────────────────────────────────
@app.command("gse-test")
def gse_test(
    ateco_l1:     str = typer.Option("DO", "--ateco-l1", help="ATECO L1 to filter PODs."),
    dayset:       str = typer.Option("weekday", help="weekday | weekend | all"),
    profile_code: str = typer.Option("PDMM", help="GSE column to compare against."),
    min_months:   int = typer.Option(12),
):
    """End-to-end GSE comparison smoke test (SQL aggregation + metrics)."""
    from data_explorer.core.gse import (
        compare_to_gse, fetch_gse_reference_profile,
        fetch_our_gse_normalised_profile,
    )
    from data_explorer.db.queries import (
        fetch_pods_by_ateco, fetch_pods_with_data_coverage,
    )

    with SessionLocal() as session:
        cov = fetch_pods_with_data_coverage(session, min_months=min_months)
        ate = fetch_pods_by_ateco(session, [ateco_l1], level=1)
        pods = cov & ate
        console.print(f"PODs selected: {len(pods):,} "
                      f"(ATECO L1={ateco_l1!r}, ≥{min_months}mo)")
        if not pods:
            console.print("[red]No PODs matched.[/red]")
            raise typer.Exit(code=1)

        console.print("[bold]Aggregating our profile in PostgreSQL…[/bold]")
        ours = fetch_our_gse_normalised_profile(session, pods, dayset=dayset)
        console.print(f"  Months covered: {ours.index.get_level_values(0).nunique()}")

        ref  = fetch_gse_reference_profile(session, profile_code)
        if ref.empty:
            console.print(f"[red]No reference data for {profile_code!r}. "
                          f"Did you run `data-explorer ingest-gse`?[/red]")
            raise typer.Exit(code=1)

        metrics = compare_to_gse(ours, ref)
        if metrics.empty:
            console.print("[yellow]No overlap between our profile and the reference.[/yellow]")
            return

        t = Table(title=f"Comparison: PoliTo ({ateco_l1}, {dayset}) vs GSE {profile_code}",
                  show_header=True, header_style="bold cyan")
        t.add_column("Month", justify="right")
        t.add_column("RMSE (pp)", justify="right")
        t.add_column("MAE (pp)",  justify="right")
        t.add_column("Bias (pp)", justify="right")
        t.add_column("Max |err| (pp)", justify="right")
        for m, r in metrics.iterrows():
            t.add_row(f"{int(m):>2d}", f"{r.rmse:.3f}", f"{r.mae:.3f}",
                      f"{r.bias:+.3f}", f"{r.max_abs_err:.3f}")
        console.print(t)


# ── arera-test ───────────────────────────────────────────────────────────────
@app.command("arera-test")
def arera_test(
    power_class: str = typer.Option("1.5–3 kW", help="Power class label."),
    market:      str = typer.Option("Tutti", help="Tipo mercato."),
    residenza:   str = typer.Option("Residente", help="Residenza."),
    day_type:    str = typer.Option("Weekday", help="Weekday / Saturday / Sunday"),
    month:       int = typer.Option(0, help="Month 1–12, or 0 for annual avg."),
    province:    str = typer.Option("Trento"),
    min_months:  int = typer.Option(12),
):
    """End-to-end ARERA comparison smoke test."""
    from data_explorer.core.arera import (
        compare_to_arera, fetch_arera_reference, fetch_our_arera_profile,
    )
    from data_explorer.db.queries import (
        fetch_pod_metadata, fetch_pods_by_ateco, fetch_pods_with_data_coverage,
    )

    # POD set: domestic + matching power class + coverage filter
    with SessionLocal() as session:
        cov = fetch_pods_with_data_coverage(session, min_months=min_months)
        domestic_l1 = "DO"
        dom = fetch_pods_by_ateco(session, [domestic_l1], level=1)
        # Filter by residenza via ateco_l2 mapping (DO.R / DO.NR)
        if residenza == "Residente":
            res_pods = fetch_pods_by_ateco(session, ["DO.R"], level=2)
        elif residenza == "Non Residente":
            res_pods = fetch_pods_by_ateco(session, ["DO.NR"], level=2)
        else:
            res_pods = dom

        candidate = cov & res_pods
        if not candidate:
            console.print("[red]No PODs match before power-class filter.[/red]")
            raise typer.Exit(1)

        # Power-class filter via measurements.potenza_contrattuale (most recent row)
        rows = session.execute(
            text(
                "SELECT DISTINCT ON (pod) pod, potenza_contrattuale "
                "FROM measurements "
                "WHERE pod = ANY(:pods) AND potenza_contrattuale IS NOT NULL "
                "ORDER BY pod, data_misura DESC"
            ),
            {"pods": list(candidate)},
        ).all()
        pc_df = pd.DataFrame(rows, columns=["pod", "potcontr"])
        pc_df["kw"] = pd.to_numeric(
            pc_df["potcontr"].astype(str).str.replace(",", ".", regex=False).str.strip(),
            errors="coerce",
        )
        pc_df = pc_df.dropna(subset=["kw"])
        pc_filters = {
            "≤ 1.5 kW": (pc_df["kw"] > 0)   & (pc_df["kw"] <= 1.5),
            "1.5–3 kW": (pc_df["kw"] > 1.5) & (pc_df["kw"] <= 3),
            "3–4.5 kW": (pc_df["kw"] > 3)   & (pc_df["kw"] <= 4.5),
            "4.5–6 kW": (pc_df["kw"] > 4.5) & (pc_df["kw"] <= 6),
            "> 6 kW":   (pc_df["kw"] > 6),
        }
        mask = pc_filters.get(power_class)
        if mask is None:
            console.print(f"[red]Unknown power class: {power_class!r}[/red]")
            raise typer.Exit(1)
        pods = set(pc_df.loc[mask, "pod"])
        console.print(f"PODs in {power_class} / {residenza}: {len(pods):,}")
        if not pods:
            raise typer.Exit(1)

        # Profiles
        ours = fetch_our_arera_profile(session, pods, day_type=day_type, month_idx=month)
        ref  = fetch_arera_reference(
            session, power_class=power_class, market=market, residenza=residenza,
            day_type=day_type, month_idx=month, province=province,
        )
        if ref.empty:
            console.print(f"[red]No reference data for the selected key. "
                          f"Did you run `data-explorer ingest-arera`?[/red]")
            raise typer.Exit(1)

        # Print profiles side-by-side
        t = Table(title=f"{power_class} | {market} | {residenza} | "
                        f"{day_type} | month={month or 'annual'}",
                  show_header=True, header_style="bold cyan")
        t.add_column("Hour", justify="right")
        t.add_column("Ours (kWh)", justify="right")
        t.add_column("ARERA (kWh)", justify="right")
        t.add_column("Δ", justify="right")
        for h in range(24):
            o, r = float(ours.get(h, 0.0)), float(ref.get(h, 0.0))
            t.add_row(f"{h:02d}", f"{o:.3f}", f"{r:.3f}", f"{o - r:+.3f}")
        console.print(t)

        metrics = compare_to_arera(ours, ref)
        m = Table(title="Comparison metrics (kWh)", show_header=True,
                  header_style="bold cyan")
        m.add_column("Metric"); m.add_column("Value", justify="right")
        for k in ("rmse", "mae", "max_abs_err", "bias"):
            v = metrics[k]
            m.add_row(k.upper(), f"{v:.4f}" if v is not None else "N/A")
        console.print(m)


# ── api ──────────────────────────────────────────────────────────────────────
@app.command("api")
def run_api(
    host: str = typer.Option(settings.backend_host),
    port: int = typer.Option(settings.backend_port),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (dev)."),
):
    """Run the FastAPI server (same as the default container command)."""
    import uvicorn
    uvicorn.run(
        "data_explorer.api.main:app",
        host=host, port=port, reload=reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    app()
