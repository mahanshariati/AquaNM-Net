from pathlib import Path

from aqua_nmnet.data.aqua_dataset import AquaPairedDataset


def _load_filename_set(
    list_file: str | Path | None,
) -> set[str] | None:
    if list_file is None:
        return None
        
    list_file = Path(list_file)
    
    if not list_file.is_file():
        raise FileNotFoundError(
            f"Dataset list file not found: {list_file}"
        )

    with list_file.open("r", encoding="utf-8") as file:
        filenames = {line.strip() for line in file if line.strip()}

    return filenames



def create_aqua_dataset(
    input_dir,
    target_dir,
    joint_transform=None,
    image_transform=None,
    exclude_list_file=None,
    include_list_file=None,
):
    exclude_names = _load_filename_set(exclude_list_file)

    include_names = _load_filename_set(include_list_file)

    return AquaPairedDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        joint_transform=joint_transform,
        transform=image_transform,
        exclude_names=exclude_names,
        include_names=include_names,
    )