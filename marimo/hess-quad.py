import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import sympy

    return (sympy,)


@app.cell
def _(sympy):
    F = sympy.Matrix(sympy.symbols("F:3:3")).reshape(3, 3)
    F
    return (F,)


@app.cell
def _(sympy):
    def hat(z: sympy.Matrix) -> sympy.Matrix:
        return sympy.Matrix(
            [
                [0, -z[2], z[1]],
                [z[2], 0, -z[0]],
                [-z[1], z[0], 0],
            ]
        )

    return (hat,)


@app.cell
def _(F, hat, sympy):
    f0, f1, f2 = (hat(F.col(i)) for i in range(3))
    z33 = sympy.zeros(3, 3)
    H3 = sympy.Matrix([[z33, -f2, f1], [f2, z33, -f0], [-f1, f0, z33]])
    H3
    return (H3,)


@app.cell
def _(sympy):
    p = sympy.Matrix(sympy.symbols("p:3:3")).reshape(3, 3)
    p
    return (p,)


@app.cell
def _(sympy):
    def vec(p: sympy.Matrix) -> list[sympy.Expr]:
        return p.T.flat()

    return (vec,)


@app.cell
def _(H3, p, sympy, vec):
    p_vec = sympy.Matrix(vec(p))
    expected = p_vec.T @ H3 @ p_vec
    expected
    return (expected,)


@app.cell
def _(sympy):
    def h6_quad(F: sympy.Matrix, p: sympy.Matrix) -> sympy.Expr:
        f0, f1, f2 = F.col(0), F.col(1), F.col(2)
        p0, p1, p2 = p.col(0), p.col(1), p.col(2)
        return (
            p0.T @ (f1.cross(p2) - f2.cross(p1))
            + p1.T @ (f2.cross(p0) - f0.cross(p2))
            + p2.T @ (f0.cross(p1) - f1.cross(p0))
        )

    return (h6_quad,)


@app.cell
def _(F, h6_quad, p):
    actual = h6_quad(F, p)
    actual
    return (actual,)


@app.cell
def _(actual, expected):
    actual - expected


if __name__ == "__main__":
    app.run()
