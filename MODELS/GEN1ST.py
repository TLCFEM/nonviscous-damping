import sympy as sp


def derive_characteristic_coefficients():
    c1, c2, c3, c4, c5, c6 = sp.symbols("c1 c2 c3 c4 c5 c6", real=True)
    lam = sp.symbols("lambda")

    char_poly = sp.collect(
        sp.expand(
            (
                lam * sp.eye(3)
                - sp.Matrix(
                    [[0, 1, 0], [-c1, 0, -c2], [c6 - c4 * c1, c5, -c3 - c4 * c2]]
                )
            ).det()
        ),
        lam,
    )

    poly_obj = sp.Poly(char_poly, lam)
    coefficients = poly_obj.all_coeffs()

    print("--- Poly Weights (Coefficients) ---")
    print(f"a3 (lambda^3 weight): {coefficients[0]}")
    print(f"a2 (lambda^2 weight): {sp.simplify(coefficients[1])}")
    print(f"a1 (lambda^1 weight): {sp.simplify(coefficients[2])}")
    print(f"a0 (lambda^0 weight): {sp.simplify(coefficients[3])}")
    print(
        sp.simplify(
            coefficients[1] * coefficients[2] - coefficients[0] * coefficients[3]
        )
    )


if __name__ == "__main__":
    derive_characteristic_coefficients()
