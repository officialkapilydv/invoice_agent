"""
CLI entry point for the Invoice AI Agent.

Commands:
  extract <pdf>               Extract and print JSON to stdout
  extract <pdf> --save        Extract and persist to SQLite
  batch <folder>              Extract all PDFs in a folder
  batch <folder> --out <dir>  Also write each result as a .json file
  correct <id> <json_path>    Submit a correction for in-context learning
  stats                       Show extraction statistics

Examples:
  python main.py extract data/sample_invoices/inv001.pdf --save
  python main.py batch data/sample_invoices --save --out results/
  python main.py correct 1 corrected.json
  python main.py stats
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.json import JSON
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from config.settings import configure_logging, validate_config
from src.agent import InvoiceAgent
from src.database import get_extraction, get_stats, init_db, save_extraction
from src.memory.feedback_loop import submit_correction
from src.schema import simplify_invoice

console = Console()


@click.group()
def cli() -> None:
    """Invoice AI Agent — extract structured data from PDF invoices."""
    configure_logging()
    init_db()


@cli.command()
@click.argument("pdf_path", type=click.Path(exists=True))
@click.option("--save", is_flag=True, default=False, help="Save result to database")
@click.option("--simple", is_flag=True, default=False, help="Print simplified output (invoice_number, vendor, line items, grand_total only)")
def extract(pdf_path: str, save: bool, simple: bool) -> None:
    """
    Extract invoice data from PDF_PATH and print as JSON.

    Full extraction always runs (validator, ChromaDB, pattern rules).
    Use --simple to display only the minimal fields instead of the full schema.
    The database always stores the FULL extraction regardless of --simple.
    """
    validate_config()
    agent = InvoiceAgent()

    try:
        result = agent.run(pdf_path)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    invoice_dict = result.invoice.model_dump(mode="json")

    if simple:
        simplified = simplify_invoice(result.invoice)
        console.print("\n[bold green]Extracted Invoice (simplified)[/bold green]")
        console.print(JSON(json.dumps(simplified.model_dump(mode="json"), indent=2, default=str)))
    else:
        console.print("\n[bold green]Extracted Invoice[/bold green]")
        console.print(JSON(json.dumps(invoice_dict, indent=2, default=str)))

    if result.invoice.validation_warnings:
        console.print("\n[bold yellow]Validation Warnings[/bold yellow]")
        for w in result.invoice.validation_warnings:
            console.print(f"  [yellow]•[/yellow] [{w.field}] {w.message}")

    if save:
        extraction_id = save_extraction(
            pdf_filename=Path(pdf_path).name,
            extracted_json=invoice_dict,          # always save the FULL extraction
            confidence=result.invoice.confidence_score,
            used_ocr=result.extraction.used_ocr,
            model=result.llm_response.model,
            prompt_tokens=result.llm_response.prompt_tokens,
            completion_tokens=result.llm_response.completion_tokens,
        )
        console.print(
            f"\n[green]Saved to database — extraction_id: {extraction_id}[/green]"
        )


@cli.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--save", is_flag=True, default=False, help="Save all results to database")
@click.option("--out", default=None, help="Directory to write per-invoice .json files")
@click.option("--skip-errors", is_flag=True, default=True, help="Continue on error instead of stopping")
@click.option("--delay", default=3, show_default=True, type=int, help="Seconds to wait between API calls (avoids rate limits)")
@click.option("--simple", is_flag=True, default=False, help="Write simplified JSON to --out folder instead of full extraction")
def batch(folder: str, save: bool, out: str | None, skip_errors: bool, delay: int, simple: bool) -> None:
    """
    Extract all PDFs in FOLDER and print a summary table.

    Full extraction always runs and is always saved to the database.
    Use --simple with --out to write minimal JSON files (invoice_number,
    vendor, line items, grand_total) instead of the full schema.
    """
    pdf_files = sorted(Path(folder).glob("*.pdf")) + sorted(Path(folder).glob("*.PDF"))
    # Deduplicate in case the OS returns both casings
    seen: set[str] = set()
    unique_pdfs: list[Path] = []
    for p in pdf_files:
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique_pdfs.append(p)

    if not unique_pdfs:
        console.print(f"[yellow]No PDF files found in {folder}[/yellow]")
        return

    validate_config()

    out_dir: Path | None = None
    if out:
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)

    agent = InvoiceAgent()

    # Tracking counters for the summary
    results: list[dict] = []   # {file, invoice_number, confidence, warnings, id, error}

    console.print(f"\n[bold]Processing {len(unique_pdfs)} PDF(s) from [cyan]{folder}[/cyan][/bold]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting...", total=len(unique_pdfs))

        for pdf in unique_pdfs:
            progress.update(task, description=f"[cyan]{pdf.name}[/cyan]")

            row: dict = {"file": pdf.name, "invoice_number": "—", "confidence": "—",
                         "warnings": 0, "id": "—", "error": None}

            try:
                result = agent.run(str(pdf))
                inv = result.invoice
                invoice_dict = inv.model_dump(mode="json")

                row["invoice_number"] = inv.invoice_number
                row["confidence"] = f"{inv.confidence_score:.2f}"
                row["warnings"] = len(inv.validation_warnings)

                if save:
                    eid = save_extraction(
                        pdf_filename=pdf.name,
                        extracted_json=invoice_dict,
                        confidence=inv.confidence_score,
                        used_ocr=result.extraction.used_ocr,
                        model=result.llm_response.model,
                        prompt_tokens=result.llm_response.prompt_tokens,
                        completion_tokens=result.llm_response.completion_tokens,
                    )
                    row["id"] = str(eid)

                if out_dir:
                    out_path = out_dir / (pdf.stem + ".json")
                    write_dict = (
                        simplify_invoice(inv).model_dump(mode="json")
                        if simple else invoice_dict
                    )
                    out_path.write_text(
                        json.dumps(write_dict, indent=2, default=str),
                        encoding="utf-8",
                    )

            except Exception as exc:
                row["error"] = str(exc)[:60]
                logging.getLogger(__name__).error("Failed on %s: %s", pdf.name, exc)
                if not skip_errors:
                    progress.stop()
                    console.print(f"[red]Stopped on error:[/red] {exc}")
                    break

            results.append(row)
            progress.advance(task)

            # Pause between calls to stay within Groq's rate limit.
            # Free tier: ~30 req/min. 3 s gap = 20 req/min — safe headroom.
            if delay > 0:
                time.sleep(delay)

    # Summary table
    table = Table(title="Batch Extraction Results", show_header=True, show_lines=True)
    table.add_column("File", style="cyan", no_wrap=True, max_width=35)
    table.add_column("Invoice #", style="white")
    table.add_column("Confidence", justify="center")
    table.add_column("Warnings", justify="center")
    if save:
        table.add_column("DB id", justify="center")
    table.add_column("Status", justify="center")

    ok = warn = fail = 0
    for r in results:
        if r["error"]:
            status = "[red]ERROR[/red]"
            fail += 1
        elif r["warnings"] > 0:
            status = f"[yellow]WARN ({r['warnings']})[/yellow]"
            warn += 1
        else:
            status = "[green]OK[/green]"
            ok += 1

        row_data = [r["file"], r["invoice_number"], r["confidence"], str(r["warnings"])]
        if save:
            row_data.append(r["id"])
        row_data.append(status)
        table.add_row(*row_data)

    console.print(table)
    console.print(
        f"\n[green]OK: {ok}[/green]  "
        f"[yellow]WARN: {warn}[/yellow]  "
        f"[red]FAIL: {fail}[/red]"
    )
    if out_dir:
        console.print(f"JSON files written to [cyan]{out_dir}[/cyan]")


@cli.command()
@click.argument("extraction_id", type=int)
@click.argument("corrected_json_path", type=click.Path(exists=True))
@click.option("--notes", default=None, help="Explain what was wrong (helps pattern learning)")
def correct(
    extraction_id: int, corrected_json_path: str, notes: str | None
) -> None:
    """
    Submit a correction for extraction EXTRACTION_ID.

    Saves to SQLite + ChromaDB (high-priority few-shot memory).
    Automatically triggers Layer 2 pattern extraction every 10 corrections.
    """
    with open(corrected_json_path, encoding="utf-8") as f:
        corrected_json: dict = json.load(f)

    try:
        result = submit_correction(
            extraction_id=extraction_id,
            corrected_json=corrected_json,
            user_notes=notes,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(f"[green]Correction saved[/green]")

    if result["changed_fields"]:
        console.print(f"  Fields corrected: [yellow]{', '.join(result['changed_fields'])}[/yellow]")

    if result["pattern_extraction_triggered"]:
        console.print(
            "[bold cyan]Pattern extraction ran — new rules added to learned_rules.json[/bold cyan]\n"
            "  Run [bold]python main.py stats[/bold] to see the rule count."
        )
    else:
        from src.database import get_stats as _get_stats
        db = _get_stats()
        remaining = 10 - (db["total_corrections"] % 10)
        console.print(
            f"  [dim]{remaining} more correction(s) until next pattern extraction[/dim]"
        )


@cli.command()
def stats() -> None:
    """Show extractions, corrections, field accuracy, and learned rules."""
    from src.memory.feedback_loop import get_field_accuracy_report
    from src.memory.pattern_library import load_rules
    from src.memory.vector_store import VectorStore

    db = get_stats()
    accuracy = get_field_accuracy_report()
    rules = load_rules()

    try:
        vs = VectorStore()
        vs_stats = vs.get_stats()
    except Exception:
        vs_stats = {"vector_extractions": "n/a", "vector_corrections": "n/a"}

    # Main stats table
    table = Table(title="Invoice Agent Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Extractions", str(db["total_extractions"]))
    table.add_row("  Digital PDF", str(db["digital_extractions"]))
    table.add_row("  OCR (scanned)", str(db["ocr_extractions"]))
    table.add_row("Total Corrections", str(db["total_corrections"]))
    table.add_row("Avg Confidence Score", str(db["avg_confidence"]))
    table.add_row("Vector DB (extractions)", str(vs_stats["vector_extractions"]))
    table.add_row("Vector DB (corrections)", str(vs_stats["vector_corrections"]))
    table.add_row("Learned Rules", str(len(rules)))

    console.print(table)

    # Field accuracy table (only if corrections exist)
    if accuracy.get("field_error_counts"):
        acc_table = Table(title="Most-Corrected Fields", show_header=True)
        acc_table.add_column("Field", style="yellow")
        acc_table.add_column("Times Wrong", justify="center", style="red")
        acc_table.add_column("Error Rate", justify="center")

        total = accuracy["total_corrections"]
        for field, count in list(accuracy["field_error_counts"].items())[:8]:
            rate = f"{count/total*100:.0f}%"
            acc_table.add_row(field, str(count), rate)

        console.print(acc_table)

    # Learned rules summary
    if rules:
        rules_table = Table(title="Learned Rules", show_header=True)
        rules_table.add_column("Trigger", style="cyan", max_width=20)
        rules_table.add_column("Rule", style="white", max_width=60)
        rules_table.add_column("Conf.", justify="center")
        rules_table.add_column("Support", justify="center")

        for r in rules:
            rules_table.add_row(
                r.get("trigger") or "(global)",
                r.get("rule", ""),
                str(r.get("confidence", "")),
                str(r.get("support_count", "")),
            )
        console.print(rules_table)


if __name__ == "__main__":
    cli()
