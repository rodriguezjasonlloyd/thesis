"""
Comprehensive Data Leakage Detection Script
Tests for common issues that cause unrealistic 100% accuracy
"""

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.table import Table

from modules.data import get_class_names, get_data_loaders

console = Console()


def compute_image_hash(image_path: Path) -> str:
    """Compute MD5 hash of image file."""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compute_perceptual_hash(image_path: Path, hash_size: int = 8) -> str:
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
        return ""


def test_duplicate_images(root: Path):
    """Test for exact duplicate images."""
    console.print(
        "\n[bold cyan]═══ Test 1: Checking for Exact Duplicates ═══[/bold cyan]"
    )

    class_names = get_class_names(root)
    all_hashes = {}
    duplicates = defaultdict(list)

    for class_name in class_names:
        class_dir = root / class_name
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                img_hash = compute_image_hash(img_path)
                if img_hash in all_hashes:
                    duplicates[img_hash].append(img_path)
                    if all_hashes[img_hash] not in duplicates[img_hash]:
                        duplicates[img_hash].insert(0, all_hashes[img_hash])
                else:
                    all_hashes[img_hash] = img_path

    if duplicates:
        console.print(
            f"[red]✗ FOUND {len(duplicates)} GROUPS OF DUPLICATE IMAGES![/red]"
        )
        for i, (hash_val, paths) in enumerate(list(duplicates.items())[:5]):
            console.print(
                f"\n[yellow]Duplicate Group {i + 1} ({len(paths)} copies):[/yellow]"
            )
            for path in paths[:3]:
                console.print(f"  - {path}")
            if len(paths) > 3:
                console.print(f"  ... and {len(paths) - 3} more")
        return False
    else:
        console.print("[green]✓ No exact duplicates found[/green]")
        return True


def test_similar_images(root: Path, threshold: int = 5):
    """Test for perceptually similar images."""
    console.print(
        "\n[bold cyan]═══ Test 2: Checking for Similar Images ═══[/bold cyan]"
    )

    class_names = get_class_names(root)
    all_phashes = {}
    similar_groups = []

    for class_name in class_names:
        class_dir = root / class_name
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                phash = compute_perceptual_hash(img_path)
                if phash:
                    all_phashes[img_path] = phash

    # Compare all pairs
    paths = list(all_phashes.keys())
    for i, path1 in enumerate(paths):
        for path2 in paths[i + 1 :]:
            # Hamming distance
            distance = sum(
                c1 != c2 for c1, c2 in zip(all_phashes[path1], all_phashes[path2])
            )
            if distance <= threshold:
                similar_groups.append((path1, path2, distance))

    if similar_groups:
        console.print(
            f"[yellow]⚠ FOUND {len(similar_groups)} PAIRS OF SIMILAR IMAGES![/yellow]"
        )
        for path1, path2, dist in similar_groups[:5]:
            console.print(f"  Hamming distance {dist}: {path1.name} ↔ {path2.name}")
        if len(similar_groups) > 5:
            console.print(f"  ... and {len(similar_groups) - 5} more pairs")
        return len(similar_groups) < 10  # Warning but not critical if few
    else:
        console.print("[green]✓ No highly similar images found[/green]")
        return True


def test_fold_overlap(root: Path, k_folds: int = 5, seed: int = 42):
    """Test for train/validation overlap across folds."""
    console.print(
        "\n[bold cyan]═══ Test 3: Checking for Train/Val Overlap ═══[/bold cyan]"
    )

    # Seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fold_loaders = get_data_loaders(
        root=root,
        k_folds=k_folds,
        batch_size=32,
        num_workers=0,  # Avoid multiprocessing issues
    )

    all_good = True
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        train_paths = {item[0] for item in train_loader.dataset._items}
        val_paths = {item[0] for item in val_loader.dataset._items}

        overlap = train_paths & val_paths

        if overlap:
            console.print(
                f"[red]✗ FOLD {fold_idx + 1}: {len(overlap)} OVERLAPPING IMAGES![/red]"
            )
            for path in list(overlap)[:3]:
                console.print(f"  - {path}")
            all_good = False
        else:
            console.print(
                f"[green]✓ Fold {fold_idx + 1}: No overlap ({len(train_paths)} train, {len(val_paths)} val)[/green]"
            )

    return all_good


def test_class_distribution(root: Path, k_folds: int = 5, seed: int = 42):
    """Test class distribution across folds."""
    console.print(
        "\n[bold cyan]═══ Test 4: Checking Class Distribution ═══[/bold cyan]"
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fold_loaders = get_data_loaders(
        root=root,
        k_folds=k_folds,
        batch_size=32,
        num_workers=0,
    )

    table = Table(title="Class Distribution per Fold")
    table.add_column("Fold", style="cyan")
    table.add_column("Split", style="magenta")
    table.add_column("Total", justify="right", style="white")
    table.add_column("Class 0", justify="right", style="green")
    table.add_column("Class 1", justify="right", style="yellow")
    table.add_column("Balance %", justify="right", style="blue")

    all_balanced = True
    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        for split_name, loader in [("Train", train_loader), ("Val", val_loader)]:
            labels = [item[1] for item in loader.dataset._items]
            class_counts = Counter(labels)
            total = len(labels)
            class_0 = class_counts[0]
            class_1 = class_counts[1]
            balance = min(class_0, class_1) / max(class_0, class_1) * 100

            table.add_row(
                str(fold_idx + 1),
                split_name,
                str(total),
                str(class_0),
                str(class_1),
                f"{balance:.1f}%",
            )

            if balance < 40:  # Less than 40% balance is problematic
                all_balanced = False

    console.print(table)

    if all_balanced:
        console.print("[green]✓ All folds have reasonable class balance[/green]")
    else:
        console.print("[yellow]⚠ Some folds have severe class imbalance[/yellow]")

    return all_balanced


def test_label_mapping(root: Path):
    """Test that label mapping is correct."""
    console.print("\n[bold cyan]═══ Test 5: Checking Label Mapping ═══[/bold cyan]")

    class_names = get_class_names(root)
    label_map = {label: index for index, label in enumerate(class_names)}

    console.print(f"Class names (sorted): {class_names}")
    console.print(f"Label mapping: {label_map}")

    # Check each class directory
    for class_name in class_names:
        class_dir = root / class_name
        files = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        console.print(
            f"  {class_name}: {len(files)} images → Label {label_map[class_name]}"
        )

    # Verify expected mapping
    if "infected" in class_names and "notinfected" in class_names:
        if label_map["infected"] == 0 and label_map["notinfected"] == 1:
            console.print(
                "[green]✓ Label mapping is as expected (infected=0, notinfected=1)[/green]"
            )
            return True
        else:
            console.print("[yellow]⚠ Label mapping differs from expected[/yellow]")
            return True  # Not necessarily wrong, just different

    console.print("[green]✓ Label mapping established[/green]")
    return True


def test_seed_reproducibility(root: Path, k_folds: int = 5):
    """Test if results are reproducible with same seed."""
    console.print(
        "\n[bold cyan]═══ Test 6: Checking Seed Reproducibility ═══[/bold cyan]"
    )

    seed = 42

    # Run 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loaders1 = get_data_loaders(
        root=root, k_folds=k_folds, batch_size=32, num_workers=0
    )
    paths1 = [[item[0] for item in loader[1].dataset._items] for loader in loaders1]

    # Run 2
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    loaders2 = get_data_loaders(
        root=root, k_folds=k_folds, batch_size=32, num_workers=0
    )
    paths2 = [[item[0] for item in loader[1].dataset._items] for loader in loaders2]

    # Compare
    if paths1 == paths2:
        console.print("[green]✓ Splits are reproducible with same seed[/green]")
        return True
    else:
        console.print(
            "[red]✗ SPLITS ARE NOT REPRODUCIBLE! Random seeding is broken![/red]"
        )
        return False


def test_image_dimensions(root: Path, sample_size: int = 20):
    """Test for unusual image dimensions or corrupted files."""
    console.print("\n[bold cyan]═══ Test 7: Checking Image Integrity ═══[/bold cyan]")

    class_names = get_class_names(root)
    dimensions = defaultdict(int)
    corrupted = []

    all_images = []
    for class_name in class_names:
        class_dir = root / class_name
        all_images.extend(class_dir.glob("*.jpg"))
        all_images.extend(class_dir.glob("*.jpeg"))
        all_images.extend(class_dir.glob("*.png"))

    # Sample images
    sample = random.sample(all_images, min(sample_size, len(all_images)))

    for img_path in sample:
        try:
            with Image.open(img_path) as img:
                dimensions[img.size] += 1
        except Exception as e:
            corrupted.append((img_path, str(e)))

    if corrupted:
        console.print(f"[red]✗ FOUND {len(corrupted)} CORRUPTED IMAGES![/red]")
        for path, error in corrupted[:3]:
            console.print(f"  {path}: {error}")
        return False

    console.print("[green]✓ All sampled images are valid[/green]")
    console.print(f"Dimension distribution (sample of {len(sample)}):")
    for dim, count in sorted(dimensions.items(), key=lambda x: -x[1])[:5]:
        console.print(f"  {dim[0]}x{dim[1]}: {count} images")

    return True


def test_data_too_easy(root: Path, sample_size: int = 10):
    """Check if images might be artificially easy to classify."""
    console.print(
        "\n[bold cyan]═══ Test 8: Checking if Data Might Be Too Easy ═══[/bold cyan]"
    )

    class_names = get_class_names(root)

    if len(class_names) != 2:
        console.print("[yellow]⚠ Not a binary classification problem[/yellow]")
        return True

    # Sample images from each class
    class_samples = {}
    for class_name in class_names:
        class_dir = root / class_name
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        class_samples[class_name] = random.sample(images, min(sample_size, len(images)))

    # Compute average pixel statistics
    stats = {}
    for class_name, paths in class_samples.items():
        means = []
        stds = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    arr = np.array(img.convert("RGB"))
                    means.append(arr.mean())
                    stds.append(arr.std())
            except:
                pass

        stats[class_name] = {
            "mean": np.mean(means) if means else 0,
            "std": np.mean(stds) if stds else 0,
        }

    console.print("Average pixel statistics per class:")
    for class_name, stat in stats.items():
        console.print(f"  {class_name}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")

    # Check if classes are suspiciously different
    if len(stats) == 2:
        classes = list(stats.keys())
        mean_diff = abs(stats[classes[0]]["mean"] - stats[classes[1]]["mean"])

        if mean_diff > 50:
            console.print(
                f"[yellow]⚠ Large brightness difference ({mean_diff:.1f}) between classes![/yellow]"
            )
            console.print("  This could make classification artificially easy")
            return False

    console.print("[green]✓ No obvious artificial patterns detected[/green]")
    return True


def run_all_tests(data_root: str = "data", k_folds: int = 5, seed: int = 42):
    """Run all data leakage tests."""
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print(
        "[bold magenta]    DATA LEAKAGE DETECTION TEST SUITE    [/bold magenta]"
    )
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")

    root = Path(data_root)

    if not root.exists():
        console.print(f"[red]Error: Data directory '{root}' does not exist![/red]")
        return

    results = {}

    # Run all tests
    results["duplicates"] = test_duplicate_images(root)
    results["similar"] = test_similar_images(root)
    results["overlap"] = test_fold_overlap(root, k_folds, seed)
    results["distribution"] = test_class_distribution(root, k_folds, seed)
    results["labels"] = test_label_mapping(root)
    results["reproducibility"] = test_seed_reproducibility(root, k_folds)
    results["integrity"] = test_image_dimensions(root)
    results["easy_data"] = test_data_too_easy(root)

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]SUMMARY[/bold cyan]")
    console.print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[green]✓ PASS[/green]" if passed_test else "[red]✗ FAIL[/red]"
        console.print(f"{test_name.replace('_', ' ').title()}: {status}")

    console.print("=" * 60)
    console.print(f"[bold]Tests Passed: {passed}/{total}[/bold]")

    if passed == total:
        console.print("\n[green]All tests passed! Data pipeline looks good.[/green]")
        console.print(
            "[yellow]If you're still getting 100% accuracy, the problem might be:[/yellow]"
        )
        console.print("  1. Dataset is genuinely very easy to classify")
        console.print("  2. Issue in model architecture or training loop")
        console.print("  3. Need to test on completely held-out external data")
    else:
        console.print(
            f"\n[red]{total - passed} test(s) failed! These issues could cause data leakage.[/red]"
        )


if __name__ == "__main__":
    run_all_tests(data_root="data", k_folds=5, seed=42)
