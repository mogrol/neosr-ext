from pathlib import Path
from typing import Any

from neosr.utils import scandir, get_root_logger
from os.path import join as path_join

def paired_paths_from_lmdb(folders: list[str], keys: list[str]) -> list[str]:
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    ::

        lq.lmdb
        ├── data.mdb
        ├── lock.mdb
        ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [lq_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [lq_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [lq_key, gt_key]. But got {len(keys)}"
    )
    lq_folder, gt_folder = folders
    lq_key, gt_key = keys

    if not (lq_folder.endswith(".lmdb") and gt_folder.endswith(".lmdb")):
        msg = (
            f"{lq_key} folder and {gt_key} folder should both in lmdb "
            f"formats. But received {lq_key}: {lq_folder}; "
            f"{gt_key}: {gt_folder}"
        )
        raise ValueError(msg)
    # ensure that the two meta_info files are the same
    with Path(Path(lq_folder) / "meta_info.txt").open(encoding="utf-8") as fin:
        lq_lmdb_keys = [line.split(".")[0] for line in fin]
    with Path(Path(gt_folder) / "meta_info.txt").open(encoding="utf-8") as fin:
        gt_lmdb_keys = [line.split(".")[0] for line in fin]
    if set(lq_lmdb_keys) != set(gt_lmdb_keys):
        msg = f"Keys in {lq_key}_folder and {gt_key}_folder are different."
        raise ValueError(msg)
    paths: list[Any] = []
    paths.extend(
        {f"{lq_key}_path": lmdb_key, f"{gt_key}_path": lmdb_key}
        for lmdb_key in sorted(lq_lmdb_keys)
    )
    return paths


def paired_paths_from_meta_info_file(
    folders: list[str], keys: list[str], meta_info_file: str
) -> list[dict[str, str]]:
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [lq_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [lq_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [lq_key, gt_key]. But got {len(keys)}"
    )
    lq_folder, gt_folder = folders
    lq_key, gt_key = keys

    with Path(meta_info_file).open(encoding="utf-8") as fin:
        gt_names = [line.strip().split(" ")[0] for line in fin]

    paths: list[dict[str, str]] = []
    for gt_name in gt_names:
        lq_path = str(Path(lq_folder))
        gt_path = str(Path(gt_folder) / gt_name)
        paths.append({f"{lq_key}_path": lq_path, f"{gt_key}_path": gt_path})
    return paths


def paired_paths_from_folder(
    folders: list[str], keys: list[str]
) -> list[dict[str, str]]:
    """Generate paired paths from folders.

    Args:
    ----
        folders (list[str]): A list of folder path. The order of list should
            be [lq_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].

    Returns:
    -------
        list[str]: Returned path list.

    """
    assert len(folders) == 2, (
        "The len of folders should be 2 with [lq_folder, gt_folder]. "
        f"But got {len(folders)}"
    )
    assert len(keys) == 2, (
        f"The len of keys should be 2 with [lq_key, gt_key]. But got {len(keys)}"
    )

    logger = get_root_logger()

    extensions = (".jpg", ".jpeg", ".png", ".webp")
    lq_folders, gt_folders = folders
    lq_key, gt_key = keys

    lq_folders = lq_folders if isinstance(lq_folders, list) else [lq_folders]
    gt_folders = gt_folders if isinstance(gt_folders, list) else [gt_folders]

    assert len(lq_folders) == len(gt_folders), f"The number of folders in dataroot_gt and dataroot_lq doesn't match."

    lq_paths = {}
    lq_paths_count = 0
    logger.info(f"Scanning lq folders")
    for folder in lq_folders:

        lq_paths[folder] = [
            path
            for path in scandir(folder, recursive=False, full_path=False)
            if path.lower().endswith(extensions)
        ]

        lq_paths_count += len(lq_paths[folder])

        logger.info(f"{len(lq_paths[folder])} images found in {folder}")

    gt_paths = {}
    gt_paths_count = 0
    logger.info(f"Scanning gt folders")
    for folder in gt_folders:
        gt_paths[folder] = [
            path
            for path in scandir(folder, recursive=False, full_path=False)
            if path.lower().endswith(extensions)
        ]

        gt_paths_count += len(gt_paths[folder])

        logger.info(f"{len(gt_paths[folder])} images found in {folder}")

    """
    lq_paths = [
        path
        for path in scandir(lq_folder, recursive=False, full_path=True)
        if path.lower().endswith(extensions)
    ]
    gt_paths = [
        path
        for path in scandir(gt_folder, recursive=False, full_path=True)
        if path.lower().endswith(extensions)
    ]
    """
    """
    assert len(lq_paths) == len(gt_paths), (
        f"{lq_key} and {gt_key} datasets have different number of images: "
        f"{len(lq_paths)}, {len(gt_paths)}."
    )
    """

    paths: list[dict[str, str]] = []
    """
    for gt_path in gt_paths:
        lq_path = gt_path.replace(gt_folder, lq_folder)
        assert lq_path in lq_paths, f"{lq_path} is not in {lq_key}_paths."
        paths.append({f"{lq_key}_path": lq_path, f"{gt_key}_path": gt_path})
    """
    lq_paths_keys = list(lq_paths.keys())
    gt_paths_keys = list(gt_paths.keys())

    logger.info(f"Verifying paths...")
    for index in range(len(lq_paths_keys)):
        lq_folder = lq_paths_keys[index]
        gt_folder = gt_paths_keys[index]

        # Use the lq filename as the "source of truth", this way we don't have to care if there are more
        # files in the gt folder than in the lq folder. As long as there's a file in the gt folder with the same
        # filename as in the lq folder we'll add it. This is useful for scenarios where the user have one HQ folder
        # with images that they use to generating a subset of LQ images. This way they don't have to copy both images
        # every time.
        for lq_file in lq_paths[lq_paths_keys[index]]:
            lq_path = path_join(lq_folder, lq_file)
            gt_path = path_join(gt_folder, lq_file)

            assert lq_file in gt_paths[gt_paths_keys[index]], f"{lq_file} is not in {gt_key}_paths."

            paths.append({f"{lq_key}_path": lq_path, f"{gt_key}_path": gt_path})

    return paths


def paths_from_folder(folder: str) -> list[str]:
    """Generate paths from folder.

    Args:
    ----
        folder (str): Folder path.

    Returns:
    -------
        list[str]: Returned path list.

    """
    paths = list(scandir(folder))
    return [Path(str(folder)) / path for path in paths]


def paths_from_lmdb(folder: str) -> list[str]:
    """Generate paths from lmdb.

    Args:
    ----
        folder (str): Folder path.

    Returns:
    -------
        list[str]: Returned path list.

    """
    if not folder.endswith(".lmdb"):
        msg = f"Folder {folder}folder should in lmdb format."
        raise ValueError(msg)
    with Path(Path(folder) / "meta_info.txt").open(encoding="utf-8") as fin:
        return [line.split(".")[0] for line in fin]
