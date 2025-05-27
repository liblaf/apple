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
    return (F,)


@app.cell
def _(F, sympy):
    f0, f1, f2 = F.col(0), F.col(1), F.col(2)
    g3 = sympy.Matrix([f1.cross(f2).T, f2.cross(f0).T, f0.cross(f1).T])
    g3


@app.cell
def _(F):
    J = F.det()
    J
    return (J,)


@app.cell
def _(F, J, sympy):
    dJdF = sympy.derive_by_array(J, F)
    dJdF


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
