import os, random
from aqua_nmnet.config import (
    LSUI_INPUT_DIR,
    LSUI_TARGET_DIR,
    TRAIN_SPLIT_FILE,
    VAL_SPLIT_FILE,
    TEST_SPLIT_FILE,
)


TRAIN_RATIO = 0.75
VAL_RATIO = 0.10


def _find_paired_filenames():
    if not LSUI_INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"LSUI input directory not found: {LSUI_INPUT_DIR}"
        )

    if not LSUI_TARGET_DIR.is_dir():
        raise FileNotFoundError(
            f"LSUI target directory not found: {LSUI_TARGET_DIR}"
        )

    input_names = {path.name for path in LSUI_INPUT_DIR.iterdir() if path.is_file()}
    target_names = {path.name for path in LSUI_TARGET_DIR.iterdir() if path.is_file()}

    paired_names = sorted(input_names & target_names)
    if not paired_names:
        raise RuntimeError(
            "No paired LSUI images were found."
        )

    return paired_names


def _write_split_file(output_file, filenames,):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        file.write(
            "\n".join(filenames)
        )
        file.write("\n")


def create_lsui_splits(seed=0, overwrite=False,):
    split_files = (
        TRAIN_SPLIT_FILE,
        VAL_SPLIT_FILE,
        TEST_SPLIT_FILE,
    )

    if (
        not overwrite
        and all(
            path.is_file()
            for path in split_files
        )
    ):
        print(
            "LSUI split files already exist. "
            "Existing splits will be used."
        )
        return

    paired_names = _find_paired_filenames()

    rng = random.Random(seed)
    rng.shuffle(paired_names)

    dataset_size = len(paired_names)

    train_size = int(TRAIN_RATIO * dataset_size)
    val_size = int(VAL_RATIO * dataset_size)
    train_end = train_size
    val_end = train_end + val_size

    train_names = paired_names[:train_end]
    val_names = paired_names[train_end:val_end]
    test_names = paired_names[val_end:]

    _write_split_file(TRAIN_SPLIT_FILE, train_names)
    _write_split_file(VAL_SPLIT_FILE, val_names)
    _write_split_file(TEST_SPLIT_FILE, test_names)

    print("LSUI dataset split created:")
    print(
        f"  Train: {len(train_names)}"
    )
    print(
        f"  Validation: {len(val_names)}"
    )
    print(
        f"  Test: {len(test_names)}"
    )


if __name__ == "__main__":
    create_lsui_splits(
        seed=0,
        overwrite=False,
    )