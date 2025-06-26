import sys
import typer

import logging

from rich.console import Console
from rich.table import Table

from .preprocessing import app_preprocessing

log = logging.getLogger(__name__)

console = Console()

app = typer.Typer(
    name="¨PrismToolBox",
    help="A CLI for the PrismToolBox library.",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
)

app.add_typer(app_preprocessing, name="preprocessing", help="Preprocessing commands of the PrismToolBox.")

@app.callback()
def main(verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity")):
    if verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level)


@app.command()
def version():
    """Show the version of PrismToolBox."""
    from prismtoolbox import __version__
    console.print(f"PrismToolBox version: [bold green]{__version__}[/bold green]")

@app.command()
def info():
    """Display detailed information about the installed package and dependencies."""
    from prismtoolbox import __version__
    
    table = Table(title="PrismToolBox Information")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    table.add_row("Version", __version__)
    
    # Check optional dependencies
    try:
        import torch
        table.add_row("Patch Embedding", f"✅ Available (PyTorch {torch.__version__})")
    except ImportError:
        table.add_row("Patch Embedding", "❌ Not available")
        
    try:
        import cellpose
        table.add_row("Nuclei Segmentation", f"✅ Available (Cellpose {cellpose.version})")
    except ImportError:
        table.add_row("Nuclei Segmentation", "❌ Not available")
    
    console.print(table)
    
    if not any([sys.modules.get(m) for m in ['torch', 'cellpose']]):
        console.print("\n[yellow]Tip: Install optional dependencies with:[/yellow]")
        console.print("  [bold]pip install prismtoolbox[seg,emb][/bold]")