"""Provide utility functions for type checking."""

from typing import Any, Type, TypedDict


def check_valid_typeddict_keys(data: Any, type: Type[TypedDict]) -> bool:
    """Check if `data` is a valid dict according to `type`.

    Checks if all required and only required or optional keys from `type` exist in
    `data`.

    Does not check the data type of values in `data`!  Full checks, including data type
    checks, should be included within the TypedDicts modules.  These checks should
    include this function.

    Parameters
    ----------
    data : dict
        Dictionary to perform key check on.
    type: typing.TypedDict
        Target TypedDict with optional and / or required keys.

    Returns
    -------
    bool
        True if all required and only required or optional keys exist in `data`, else
        False.
    """
    if not isinstance(data, dict):
        return False

    # Note: Pyright does not seem to know __required_keys__ and __optional_keys__,
    # which is why the "type: ignore" are here.
    return check_dict(
        data,
        type.__required_keys__,  # type: ignore
        type.__optional_keys__,  # type: ignore
    )


def check_dict(data: dict, required_keys: list[str], optional_keys: list[str]):
    """Check if `data` has all required and only required or optional keys.

    Parameters
    ----------
    data : dict
        Dictionary to perform key check on.
    required_keys : list of str
        Keys that must be present in `data`.
    optional_keys : list of str
        Keys that may be present in `data`, but are not required.

    Returns
    -------
    bool
        True if all required and only required or optional keys exist in `data`, else
        False.
    """
    # Check if required keys exist in data
    for key in required_keys:
        if key not in data:
            print("[check_dict] missing key:", key)
            return False

    # Check that only required & optional keys exist in data
    for key in data:
        if key not in required_keys and key not in optional_keys:
            print(
                f"[check_dict] non allowed key: {key}. Required keys:",
                f"{required_keys}, optional keys: {optional_keys}.",
            )
            return False

    return True
