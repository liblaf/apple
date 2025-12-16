import jax.numpy as jnp
from jaxtyping import Array, Integer


def group_indices(
    arr: Integer[Array, " N"],
) -> dict[int, Integer[Array, " group_size"]]:
    sort_idx: Integer[Array, " N"] = jnp.argsort(arr)
    unique_id: Integer[Array, " n_groups"]
    first_occurrence: Integer[Array, " group_size"]
    unique_id, first_occurrence = jnp.unique(arr[sort_idx], return_index=True)
    groups: list[Integer[Array, " group_size"]] = jnp.split(
        sort_idx, first_occurrence[1:]
    )
    index_to_group: dict[int, Integer[Array, " group_size"]] = dict(
        zip(unique_id.tolist(), groups, strict=True)
    )
    return index_to_group
