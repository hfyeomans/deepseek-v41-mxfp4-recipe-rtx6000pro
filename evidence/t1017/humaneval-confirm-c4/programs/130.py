def tri(n):
    """Return the first n + 1 numbers of the Tribonacci sequence as defined."""
    if n == 0:
        return [1]
    result = [1, 3]
    if n == 1:
        return result
    # We need to compute values up to index n.
    # For even index i: tri(i) = 1 + i/2
    # For odd index i (i >= 3): tri(i) = tri(i-1) + tri(i-2) + tri(i+1)
    # Note: tri(i) for odd i depends on tri(i+1), which is even and directly computable.
    for i in range(2, n + 1):
        if i % 2 == 0:
            result.append(1 + i // 2)
        else:
            # tri(i) = tri(i-1) + tri(i-2) + tri(i+1)
            # tri(i+1) is even, so it's 1 + (i+1)//2
            next_even = 1 + (i + 1) // 2
            result.append(result[i - 1] + result[i - 2] + next_even)
    return result


def check(candidate):

    # Check some simple cases
    
    assert candidate(3) == [1, 3, 2.0, 8.0]
    assert candidate(4) == [1, 3, 2.0, 8.0, 3.0]
    assert candidate(5) == [1, 3, 2.0, 8.0, 3.0, 15.0]
    assert candidate(6) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0]
    assert candidate(7) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0]
    assert candidate(8) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0, 5.0]
    assert candidate(9) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0, 5.0, 35.0]
    assert candidate(20) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0, 5.0, 35.0, 6.0, 48.0, 7.0, 63.0, 8.0, 80.0, 9.0, 99.0, 10.0, 120.0, 11.0]

    # Check some edge cases that are easy to work out by hand.
    assert candidate(0) == [1]
    assert candidate(1) == [1, 3]


check(tri)
