import math


def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def find_zero(xs: list):
    """ xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
    Moreover, find_zero only takes list xs having even number of coefficients
    and largest non zero coefficient as it guarantees
    a solution.
    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
    -0.5
    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3
    1.0
    """
    # Find a bracket [lo, hi] where poly changes sign.
    # Since the leading coefficient is nonzero and degree is odd (even number
    # of coefficients), poly(x) -> +inf as x -> +inf and -> -inf as x -> -inf
    # (or vice versa), so a sign change exists.
    lo, hi = -1.0, 1.0
    while poly(xs, lo) * poly(xs, hi) > 0:
        lo *= 2.0
        hi *= 2.0

    # Bisection method
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if poly(xs, lo) * poly(xs, mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0




METADATA = {}


def check(candidate):
    import math
    import random
    rng = random.Random(42)
    import copy
    for _ in range(100):
        ncoeff = 2 * rng.randint(1, 4)
        coeffs = []
        for _ in range(ncoeff):
            coeff = rng.randint(-10, 10)
            if coeff == 0:
                coeff = 1
            coeffs.append(coeff)
        solution = candidate(copy.deepcopy(coeffs))
        assert math.fabs(poly(coeffs, solution)) < 1e-4



check(find_zero)
