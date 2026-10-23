def minPath(grid, k):
    n = len(grid)
    # Find the position of the smallest value to start
    start_val = min(min(row) for row in grid)
    start_pos = None
    for i in range(n):
        for j in range(n):
            if grid[i][j] == start_val:
                start_pos = (i, j)
                break
        if start_pos:
            break

    # Greedy approach: at each step, choose the neighbor with the smallest value.
    # But greedy alone may not always be optimal for lexicographic minimality.
    # However, since all values are distinct and we want lexicographically smallest,
    # we can do a BFS/DP approach: for each cell and each remaining length, compute
    # the lexicographically smallest path.

    from functools import lru_cache

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    @lru_cache(maxsize=None)
    def best(i, j, rem):
        # returns tuple: lexicographically smallest path of length rem starting at (i,j)
        if rem == 1:
            return (grid[i][j],)
        best_path = None
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                sub = best(ni, nj, rem - 1)
                candidate = (grid[i][j],) + sub
                if best_path is None or candidate < best_path:
                    best_path = candidate
        return best_path

    # Find the overall minimum path starting from any cell
    overall = None
    for i in range(n):
        for j in range(n):
            p = best(i, j, k)
            if overall is None or p < overall:
                overall = p

    return list(overall)


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
