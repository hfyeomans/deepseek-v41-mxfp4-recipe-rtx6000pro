def starts_one_ends(n):
    """
    Given a positive integer n, return the count of the numbers of n-digit
    positive integers that start or end with 1.
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    # Count numbers that start with 1: first digit fixed as 1, remaining n-1 digits free
    start_with_one = 9 ** (n - 1)
    # Count numbers that end with 1: last digit fixed as 1, first digit 1-9, middle n-2 digits free
    end_with_one = 9 * 9 ** (n - 2)
    # Count numbers that both start and end with 1: first and last fixed, middle n-2 digits free
    both = 9 ** (n - 2)
    return start_with_one + end_with_one - both


def check(candidate):

    # Check some simple cases
    assert True, "This prints if this assert fails 1 (good for debugging!)"
    assert candidate(1) == 1
    assert candidate(2) == 18
    assert candidate(3) == 180
    assert candidate(4) == 1800
    assert candidate(5) == 18000

    # Check some edge cases that are easy to work out by hand.
    assert True, "This prints if this assert fails 2 (also good for debugging!)"



check(starts_one_ends)
