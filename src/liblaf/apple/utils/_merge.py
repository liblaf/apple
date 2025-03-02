from collections.abc import Mapping


def clone(obj: Mapping | None) -> dict:
    result: dict = {}
    if obj is None:
        return result
    for key, value in obj.items():
        if isinstance(value, Mapping):
            result[key] = clone(value)
        else:
            result[key] = value
    return result


def merge(a: Mapping | None, b: Mapping | None) -> Mapping:
    result: dict = clone(a)
    if b is None:
        return result
    for key, value in b.items():
        if isinstance(value, Mapping) and key in result:
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result
