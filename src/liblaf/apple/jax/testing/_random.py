import hypothesis.strategies as st


def seed() -> st.SearchStrategy[int]:
    return st.integers(min_value=-(2**31), max_value=2**31 - 1)
