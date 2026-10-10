def largest_prime_factor(n: int):
    """Return the largest prime factor of n. Assume n > 1 and is not a prime.
    >>> largest_prime_factor(13195)
    29
    >>> largest_prime_factor(2048)
    2
    """
    largest = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            largest = d
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        largest = n
    return largest




METADATA = {}


def check(candidate):
    assert candidate(15) == 5
    assert candidate(27) == 3
    assert candidate(63) == 7
    assert candidate(330) == 11
    assert candidate(13195) == 29



check(largest_prime_factor)
