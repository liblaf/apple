import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import sympy
    return (sympy,)


@app.cell
def _(sympy):
    dh_dX = sympy.Matrix(sympy.symbols("h:4:3", real=True)).reshape(4, 3)
    dh_dX
    return (dh_dX,)


@app.cell
def _(sympy):
    def vec33(F: sympy.Matrix) -> list[sympy.Expr]:
        return F.T.flat()
    return (vec33,)


@app.cell
def _(sympy):
    def unvec33(F: sympy.Matrix | list[sympy.Expr]) -> sympy.Matrix:
        F = sympy.Matrix(F)
        return F.reshape(3, 3).T
    return (unvec33,)


@app.cell
def _(sympy):
    def vec43(x: sympy.Matrix) -> list[sympy.Expr]:
        return x.flat()
    return (vec43,)


@app.cell
def _(sympy):
    def unvec43(x: sympy.Matrix | list[sympy.Expr]) -> sympy.Matrix:
        x = sympy.Matrix(x)
        return x.reshape(4, 3)
    return (unvec43,)


@app.cell
def _(sympy):
    x = sympy.Matrix(sympy.symbols("u:4:3"), real=True).reshape(4, 3)
    x
    return (x,)


@app.cell
def _(dh_dX, x):
    F = x.T @ dh_dX
    F
    return (F,)


@app.cell
def _(F, unvec33, vec33):
    unvec33(vec33(F))
    return


@app.cell
def _(F, sympy, vec33, vec43, x):
    dFdx_T = sympy.derive_by_array(vec33(F), vec43(x))
    dFdx_T = sympy.Matrix(dFdx_T)
    dFdx = dFdx_T.T
    dFdx
    return (dFdx,)


@app.cell
def _(dFdx, sympy, unvec33, vec43):
    p = sympy.Matrix(sympy.symbols("p:4:3")).reshape(4, 3)
    dFdx_p = dFdx @ sympy.Matrix(vec43(p))
    dFdx_p = unvec33(dFdx_p)
    dFdx_p
    return dFdx_p, p


@app.cell
def _(dFdx_p, dh_dX, p):
    p.T @ dh_dX - dFdx_p
    return


@app.cell
def _(dFdx, sympy, unvec43, vec33):
    q = sympy.Matrix(sympy.symbols("q:3:3")).reshape(3, 3)
    dFdxT_p = dFdx.T @ sympy.Matrix(vec33(q))
    dFdxT_p = unvec43(dFdxT_p)
    dFdxT_p
    return dFdxT_p, q


@app.cell
def _(dFdxT_p, dh_dX, q):
    dh_dX @ q.T - dFdxT_p
    return


@app.cell
def _(dFdx):
    dFdx.T @ dFdx
    return


if __name__ == "__main__":
    app.run()
