import shutil
from aqua_nmnet.config import (
    LSUI_INPUT_DIR,
    LSUI_TARGET_DIR,
    SEARCH_INPUT_DIR,
    SEARCH_TARGET_DIR,
    VAL_SPLIT_FILE,
)

def load_split_entries(split_file):
    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found: {split_file}"
        )

    with split_file.open("r", encoding="utf-8") as file:
        filenames = [line.strip() for line in file if line.strip()]

    if not filenames:
        raise ValueError(
            f"No filenames found in split file: {split_file}"
        )
    return filenames


def prepare_search_subset():
    SEARCH_INPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    SEARCH_TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filenames = load_split_entries(
        VAL_SPLIT_FILE
    )

    for filename in filenames:
        input_source = LSUI_INPUT_DIR / filename
        target_source = LSUI_TARGET_DIR / filename

        if not input_source.is_file():
            raise FileNotFoundError(
                f"Input image not found: {input_source}"
            )
        if not target_source.is_file():
            raise FileNotFoundError(
                f"Target image not found: {target_source}"
            )

        input_destination = SEARCH_INPUT_DIR / filename
        target_destination = SEARCH_TARGET_DIR / filename

        shutil.copy2(
            input_source,
            input_destination,
        )
        shutil.copy2(
            target_source,
            target_destination,
        )

    print(
        f"Prepared {len(filenames)} image pairs "
        f"for architecture search."
    )


if __name__ == "__main__":
    prepare_search_subset()