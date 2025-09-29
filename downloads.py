import gdown
import os
from pathlib import Path
import tarfile
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

url_validation = "https://drive.google.com/uc?id=1978YlZh_wihgJfQCQW6N9dmsWVof4xYr"
url_training = "https://drive.google.com/uc?id=1CO-5lhfx-hUbEKVguRtOT_RUgU8HK7iX"

outpath = Path("assets")
outpath.mkdir(parents=True, exist_ok=True)
validation_path = outpath / "validation.tar.gz"
training_path = outpath / "training.tar.gz"

if not validation_path.exists():
    console.print("[blue]Downloading validation dataset...[/blue]")
    gdown.download(url_validation, str(validation_path), quiet=False)
    console.print("[blue]Extracting validation dataset...[/blue]")
    with tarfile.open(validation_path, "r:gz") as tar:
        tar.extractall(path=outpath / "validation")
    console.print("[green]Validation dataset ready[/green]")

if not training_path.exists():
    console.print("[blue]Downloading training dataset...[/blue]")
    gdown.download(url_training, str(training_path), quiet=False)
    console.print("[blue]Extracting training dataset...[/blue]")
    with tarfile.open(training_path, "r:gz") as tar:
        tar.extractall(path=outpath / "training")
    console.print("[green]Training dataset ready[/green]")

console.print("[green]All datasets ready[/green]")
