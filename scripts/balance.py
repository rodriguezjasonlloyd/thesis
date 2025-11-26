"""
Dataset Balancing Script
Undersamples majority class to achieve desired ratio
"""

import random
import shutil
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def count_images(class_dir: Path) -> int:
    """Count images in a directory."""
    return len(
        list(class_dir.glob("*.jpg"))
        + list(class_dir.glob("*.jpeg"))
        + list(class_dir.glob("*.png"))
    )


def balance_dataset(
    root: Path,
    target_ratio: tuple[int, int] = (60, 40),
    minority_class: str = "noninfected",
    majority_class: str = "infected",
    seed: int = 42,
    backup: bool = True,
):
    """
    Balance dataset by undersampling majority class.

    Args:
        root: Root data directory
        target_ratio: (majority%, minority%) e.g., (60, 40) means 60:40 ratio
        minority_class: Name of minority class directory
        majority_class: Name of majority class directory
        seed: Random seed for reproducibility
        backup: Whether to backup removed images
    """
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]       DATASET BALANCING TOOL       [/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    minority_dir = root / minority_class
    majority_dir = root / majority_class

    # Check directories exist
    if not minority_dir.exists():
        console.print(f"[red]Error: Directory '{minority_dir}' not found![/red]")
        return
    if not majority_dir.exists():
        console.print(f"[red]Error: Directory '{majority_dir}' not found![/red]")
        return

    # Count current images
    minority_count = count_images(minority_dir)
    majority_count = count_images(majority_dir)

    console.print("\n[cyan]Current Dataset:[/cyan]")
    console.print(f"  {minority_class}: {minority_count} images")
    console.print(f"  {majority_class}: {majority_count} images")
    console.print(
        f"  Current ratio: {majority_count}:{minority_count} "
        f"({majority_count / (majority_count + minority_count) * 100:.1f}%:{minority_count / (majority_count + minority_count) * 100:.1f}%)"
    )

    # Calculate target majority count
    # If ratio is 60:40 and minority has 52 images
    # Then majority should have: 52 * (60/40) = 78 images
    majority_ratio, minority_ratio = target_ratio
    target_majority = int(minority_count * (majority_ratio / minority_ratio))

    console.print("\n[cyan]Target Dataset:[/cyan]")
    console.print(f"  {minority_class}: {minority_count} images (unchanged)")
    console.print(f"  {majority_class}: {target_majority} images")
    console.print(
        f"  Target ratio: {target_majority}:{minority_count} "
        f"({majority_ratio}%:{minority_ratio}%)"
    )

    if target_majority >= majority_count:
        console.print(
            f"\n[yellow]⚠ Target ({target_majority}) >= current ({majority_count})[/yellow]"
        )
        console.print("[yellow]No undersampling needed![/yellow]")
        return

    to_remove = majority_count - target_majority
    console.print(
        f"\n[yellow]Will remove {to_remove} images from {majority_class}[/yellow]"
    )

    # Confirm
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != "yes":
        console.print("[red]Aborted.[/red]")
        return

    # Get all majority class images
    random.seed(seed)
    all_images = (
        list(majority_dir.glob("*.jpg"))
        + list(majority_dir.glob("*.jpeg"))
        + list(majority_dir.glob("*.png"))
    )
    random.shuffle(all_images)

    # Select images to remove
    images_to_remove = all_images[:to_remove]

    # Backup if requested
    if backup:
        backup_dir = root.parent / f"{root.name}_balanced_backup" / majority_class
        backup_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"\n[cyan]Backing up {to_remove} images to {backup_dir}[/cyan]")

        for img_path in images_to_remove:
            dest = backup_dir / img_path.name
            shutil.move(str(img_path), str(dest))
    else:
        console.print(f"\n[yellow]Deleting {to_remove} images (no backup)[/yellow]")
        for img_path in images_to_remove:
            img_path.unlink()

    # Verify
    console.print("\n[bold cyan]Verification:[/bold cyan]")
    final_minority = count_images(minority_dir)
    final_majority = count_images(majority_dir)

    table = Table(title="Final Dataset")
    table.add_column("Class", style="cyan")
    table.add_column("Images", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")

    total = final_minority + final_majority
    table.add_row(
        minority_class,
        str(final_minority),
        f"{final_minority / total * 100:.1f}%",
    )
    table.add_row(
        majority_class,
        str(final_majority),
        f"{final_majority / total * 100:.1f}%",
    )
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total}[/bold]",
        "[bold]100%[/bold]",
    )

    console.print(table)

    console.print("\n[green]✓ Dataset balanced successfully![/green]")
    if backup:
        console.print(f"[cyan]Backup saved to: {backup_dir.parent}[/cyan]")


def main():
    """Main entry point."""
    root = Path("data")

    if not root.exists():
        console.print(f"[red]Error: Directory '{root}' not found![/red]")
        return

    # Balance to 60:40 ratio (60% infected, 40% noninfected)
    balance_dataset(
        root=root,
        target_ratio=(60, 40),  # (majority%, minority%)
        minority_class="noninfected",
        majority_class="infected",
        seed=42,
        backup=True,  # Set to False to delete instead of backup
    )


if __name__ == "__main__":
    main()
