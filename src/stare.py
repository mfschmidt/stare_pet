import os
import pathlib


def stare(
    out_path=".",
    **kwargs,
):
    """ The stare function validates the execution context,
        then orchestrates the entire STARE pipeline.

    :param out_path: The directory to which output will be written,
                     defaults to "."
    :type out_path: str, os.Path
    :return: 0 if successful, error code if not
    :rtype: int
    """

    # Validate out_path argument
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        print(f"Found out_path '{out_path}'")
    else:
        print(f"out_path '{out_path}' does not exist. creating it...")
        os.makedirs(out_path, exist_ok=True)
        if not out_path.exists():
            raise FileExistsError(f"out_path '{out_path}' does not exist and could not be created.")

    for k, v in kwargs:
        print(k, v)

    return 0
