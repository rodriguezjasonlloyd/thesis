"""
Dataset Deduplication Script
Removes exact duplicates and highly similar images
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import track

console = Console()


def compute_image_hash(image_path: Path) -> str:
    """Compute MD5 hash of image file."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_perceptual_hash(image_path: Path, hash_size: int = 32) -> str:
    """Compute perceptual hash to detect similar images."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize(
                (hash_size, hash_size), Image.Resampling.LANCZOS
            )
            pixels = np.array(img).flatten()
            avg = pixels.mean()
            return "".join("1" if p > avg else "0" for p in pixels)
    except Exception as e:
        console.print(f"[red]Error hashing {image_path}: {e}[/red]")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculate Hamming distance between two hashes."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def deduplicate_exact(root: Path, backup: bool = True) -> dict:
    """Remove exact duplicate images."""
    console.print("\n[bold cyan]Step 1: Removing Exact Duplicates[/bold cyan]")

    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    all_images = []

    for class_dir in class_dirs:
        for ext in [".jpg", ".jpeg", ".png"]:
            all_images.extend(class_dir.glob(f"*{ext}"))

    console.print(f"Scanning {len(all_images)} images...")

    hash_map = {}
    duplicates = []

    for img_path in track(all_images, description="Computing hashes"):
        img_hash = compute_image_hash(img_path)

        if img_hash in hash_map:
            duplicates.append(img_path)
            console.print(
                f"[yellow]Duplicate: {img_path} (same as {hash_map[img_hash]})[/yellow]"
            )
        else:
            hash_map[img_hash] = img_path

    # Backup and remove
    if duplicates and backup:
        backup_dir = root.parent / f"{root.name}_duplicates_backup"
        backup_dir.mkdir(exist_ok=True)
        console.print(
            f"\n[cyan]Backing up {len(duplicates)} duplicates to {backup_dir}[/cyan]"
        )

        for dup_path in duplicates:
            dest = backup_dir / dup_path.parent.name / dup_path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.move(str(dup_path), str(dest))

    console.print(f"[green]✓ Removed {len(duplicates)} exact duplicates[/green]")

    return {"removed": len(duplicates), "remaining": len(all_images) - len(duplicates)}


def deduplicate_similar(root: Path, threshold: int = 5, backup: bool = True) -> dict:
    """Remove highly similar images based on perceptual hashing."""
    console.print("\n[bold cyan]Step 2: Removing Similar Images[/bold cyan]")
    console.print(f"Hamming distance threshold: {threshold} (lower = more strict)")

    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    all_images = []

    for class_dir in class_dirs:
        for ext in [".jpg", ".jpeg", ".png"]:
            all_images.extend(class_dir.glob(f"*{ext}"))

    console.print(f"Computing perceptual hashes for {len(all_images)} images...")

    # Compute all hashes
    phash_map = {}
    for img_path in track(all_images, description="Hashing"):
        phash = compute_perceptual_hash(img_path)
        if phash:
            phash_map[img_path] = phash

    console.print("Finding similar pairs (this may take a while)...")

    # Build similarity graph
    paths = list(phash_map.keys())
    similar_pairs = []

    for i in track(range(len(paths)), description="Comparing"):
        for j in range(i + 1, len(paths)):
            distance = hamming_distance(phash_map[paths[i]], phash_map[paths[j]])
            if distance <= threshold:
                similar_pairs.append((i, j, distance))

    console.print(f"Found {len(similar_pairs)} similar pairs")

    # Build adjacency list for connected components
    from collections import defaultdict

    graph = defaultdict(set)
    for i, j, _ in similar_pairs:
        graph[i].add(j)
        graph[j].add(i)

    # Find connected components using iterative DFS (avoid recursion limit)
    visited = set()
    similar_groups = []

    for i in range(len(paths)):
        if i not in visited and i in graph:
            # Iterative DFS using stack
            component = []
            stack = [i]

            while stack:
                node = stack.pop()
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                # Add unvisited neighbors to stack
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)

            similar_groups.append([paths[idx] for idx in component])

    console.print(
        f"\n[yellow]Found {len(similar_groups)} groups of similar images[/yellow]"
    )

    # For each group, keep the first image and remove the rest
    to_remove = set()
    for group in similar_groups:
        # Keep first, remove rest
        to_remove.update(group[1:])

    removal_pct = (len(to_remove) / len(all_images)) * 100
    console.print(
        f"[yellow]Total images to remove: {len(to_remove)} ({removal_pct:.1f}% of dataset)[/yellow]"
    )

    if removal_pct > 50:
        console.print(
            "[red]⚠ WARNING: This will remove more than half your dataset![/red]"
        )
        console.print("[red]Consider using a stricter threshold (lower number)[/red]")
        response = input("\nContinue anyway? (yes/no): ").strip().lower()
        if response != "yes":
            console.print("[red]Aborted deduplication.[/red]")
            return {"groups": 0, "removed": 0, "remaining": len(all_images)}

    # Show some examples
    for i, group in enumerate(similar_groups[:5]):
        console.print(f"\n[cyan]Similar Group {i + 1} ({len(group)} images):[/cyan]")
        console.print(f"  [green]KEEPING:[/green] {group[0].name}")
        for path in group[1:4]:
            console.print(f"  [red]REMOVING:[/red] {path.name}")
        if len(group) > 4:
            console.print(f"  ... and {len(group) - 4} more to remove")

    if len(similar_groups) > 5:
        console.print(f"\n... and {len(similar_groups) - 5} more groups")

    # Backup and remove
    if to_remove and backup:
        backup_dir = root.parent / f"{root.name}_similar_backup"
        backup_dir.mkdir(exist_ok=True)
        console.print(
            f"\n[cyan]Backing up {len(to_remove)} similar images to {backup_dir}[/cyan]"
        )

        for img_path in track(to_remove, description="Backing up"):
            dest = backup_dir / img_path.parent.name / img_path.name
            dest.parent.mkdir(exist_ok=True)
            shutil.move(str(img_path), str(dest))

    console.print(f"[green]✓ Removed {len(to_remove)} similar images[/green]")

    return {
        "groups": len(similar_groups),
        "removed": len(to_remove),
        "remaining": len(all_images) - len(to_remove),
    }


def verify_cleanup(root: Path):
    """Verify the cleaned dataset."""
    console.print("\n[bold cyan]Step 3: Verifying Cleaned Dataset[/bold cyan]")

    class_dirs = [d for d in root.iterdir() if d.is_dir()]

    for class_dir in class_dirs:
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        console.print(f"  {class_dir.name}: {len(images)} images")

    total = sum(
        len(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png")))
        for d in class_dirs
    )
    console.print(f"\n[green]Total remaining: {total} images[/green]")


def main():
    """Main deduplication workflow."""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print(
        "[bold magenta]       DATASET DEDUPLICATION TOOL       [/bold magenta]"
    )
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    root = Path("data")

    if not root.exists():
        console.print(f"[red]Error: Directory '{root}' not found![/red]")
        return

    console.print("\n[yellow]⚠ This will modify your dataset![/yellow]")
    console.print(f"Original data location: {root.absolute()}")
    console.print(f"Backups will be created in: {root.parent.absolute()}")

    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != "yes":
        console.print("[red]Aborted.[/red]")
        return

    # Step 1: Remove exact duplicates
    exact_stats = deduplicate_exact(root, backup=True)

    # Step 2: Remove similar images
    # You can adjust threshold: lower = more strict
    # 5 = very similar, 10 = somewhat similar, 15 = loosely similar
    similar_stats = deduplicate_similar(root, threshold=5, backup=True)

    # Step 3: Verify
    verify_cleanup(root)

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]DEDUPLICATION COMPLETE[/bold cyan]")
    console.print("=" * 60)
    console.print(f"Exact duplicates removed: {exact_stats['removed']}")
    console.print(f"Similar images removed: {similar_stats['removed']}")
    console.print(f"Total removed: {exact_stats['removed'] + similar_stats['removed']}")
    console.print(f"Images remaining: {similar_stats['remaining']}")
    console.print(
        "\n[green]You can now retrain your model with the cleaned dataset![/green]"
    )
    console.print(
        "\n[yellow]Note: If results still seem wrong, you can restore from backups.[/yellow]"
    )


if __name__ == "__main__":
    main()
