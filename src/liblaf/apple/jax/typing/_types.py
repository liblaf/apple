from jaxtyping import Array, Float, Integer

type Scalar = Float[Array, ""]
type UpdatesData = Float[Array, "Any ..."]
type UpdatesIndex = Integer[Array, " Any"]
type Updates = tuple[UpdatesData, UpdatesIndex]
type Vector = Float[Array, "*DoF"]
