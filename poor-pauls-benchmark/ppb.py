"""
Poor Paul's Benchmark (PPB) — CLI entry point.

An automated evaluation framework for local LLM inference
powered by llama.cpp's llama-bench.
"""

import logging

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ---------------------------------------------------------------------------
# Rich console & logging setup
# ---------------------------------------------------------------------------

ppb_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "hw": "bold magenta",  # hardware specs
    }
)

console = Console(theme=ppb_theme)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
log = logging.getLogger("ppb")

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ppb",
    help="Poor Paul's Benchmark — find the absolute limit of your local AI hardware.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def download(
    repo_id: str = typer.Argument(..., help="Hugging Face repo ID (e.g. QuantFactory/Meta-Llama-3-8B-Instruct-GGUF)"),
    filename: str = typer.Argument(..., help='Glob pattern for the GGUF file (e.g. "*Q4_K_M.gguf")'),
) -> None:
    """Download a GGUF model from Hugging Face Hub."""
    console.print(f"[info]Downloading[/info] [bold]{filename}[/bold] from [bold]{repo_id}[/bold] …")
    log.info("download command — not yet implemented")


@app.command()
def sweep(
    config: str = typer.Argument(..., help="Path to a TOML sweep configuration file"),
) -> None:
    """Run a declarative parameter sweep using llama-bench."""
    console.print(f"[info]Starting sweep[/info] with config [bold]{config}[/bold] …")
    log.info("sweep command — not yet implemented")


@app.command(name="auto-limit")
def auto_limit(
    model: str = typer.Option(..., "--model", "-m", help="Path to the GGUF model file"),
    min_ctx: int = typer.Option(2048, "--min-ctx", help="Minimum context length to probe"),
    max_ctx: int = typer.Option(128000, "--max-ctx", help="Maximum context length to probe"),
) -> None:
    """Binary-search for the maximum context window before OOM."""
    console.print(
        f"[info]Auto-limit[/info] probing [hw]{model}[/hw] "
        f"between [bold]{min_ctx}[/bold] and [bold]{max_ctx}[/bold] …"
    )
    log.info("auto-limit command — not yet implemented")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
