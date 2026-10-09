def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    factor = 2
    last_factor = 1
    while factor * factor <= n:
        if n % factor == 0:
            last_factor = factor
            while n % factor == 0:
                n //= factor
        factor += 1 if factor == 2 else 2
    if n > 1:
        last_factor = n
    return last_factor




METADATA = {}


def check(candidate):
    assert candidate(15) == 5
    assert candidate(27) == 3
    assert candidate(63) == 7
    assert candidate(330) == 11
    assert candidate(13195) == 29



check(largest_prime_factor)
