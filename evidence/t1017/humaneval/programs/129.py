def minPath(grid, k):
    n = len(grid)
    # Find the position of value 1
    start = None
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 1:
                start = (i, j)
                break
        if start:
            break

    # Greedy: always move to the smallest neighbor (including staying? no, must move)
    # Actually we need lexicographically smallest sequence of length k.
    # Since 1 is the smallest value, the optimal path must start at 1.
    # Then at each step, choose the neighbor with the smallest value.
    # But careful: we need to consider that ties in value don't happen (all distinct),
    # so greedy works: at each step pick the smallest neighbor value.
    # However, we must verify greedy is optimal. Since all values distinct and we want
    # lexicographically smallest, picking smallest next value is always optimal.
    # But wait: could picking a slightly larger value now lead to smaller later?
    # No, because lexicographic comparison prioritizes earlier positions.
    # So greedy is correct.

    path = [1]
    r, c = start
    for _ in range(k - 1):
        best = None
        best_val = float('inf')
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] < best_val:
                    best_val = grid[nr][nc]
                    best = (nr, nc)
        r, c = best
        path.append(best_val)

    return path


def check(candidate):

    # Check some simple cases
    print
    assert candidate([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3) == [1, 2, 1]
    assert candidate([[5, 9, 3], [4, 1, 6], [7, 8, 2]], 1) == [1]
    assert candidate([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], 4) == [1, 2, 1, 2]
    assert candidate([[6, 4, 13, 10], [5, 7, 12, 1], [3, 16, 11, 15], [8, 14, 9, 2]], 7) == [1, 10, 1, 10, 1, 10, 1]
    assert candidate([[8, 14, 9, 2], [6, 4, 13, 15], [5, 7, 1, 12], [3, 10, 11, 16]], 5) == [1, 7, 1, 7, 1]
    assert candidate([[11, 8, 7, 2], [5, 16, 14, 4], [9, 3, 15, 6], [12, 13, 10, 1]], 9) == [1, 6, 1, 6, 1, 6, 1, 6, 1]
    assert candidate([[12, 13, 10, 1], [9, 3, 15, 6], [5, 16, 14, 4], [11, 8, 7, 2]], 12) == [1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6]
    assert candidate([[2, 7, 4], [3, 1, 5], [6, 8, 9]], 8) == [1, 3, 1, 3, 1, 3, 1, 3]
    assert candidate([[6, 1, 5], [3, 8, 9], [2, 7, 4]], 8) == [1, 5, 1, 5, 1, 5, 1, 5]

    # Check some edge cases that are easy to work out by hand.
    assert candidate([[1, 2], [3, 4]], 10) == [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    assert candidate([[1, 3], [3, 2]], 10) == [1, 3, 1, 3, 1, 3, 1, 3, 1, 3]



check(minPath)
