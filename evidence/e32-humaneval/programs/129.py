def minPath(grid, k):
    n = len(grid)
    # Map value -> position
    pos = {}
    for i in range(n):
        for j in range(n):
            pos[grid[i][j]] = (i, j)
    
    # Greedy: at each step, from current cell, choose the neighbor (or stay? no, must move)
    # Actually we can start anywhere. To minimize lexicographically, we want the smallest
    # possible first value, then smallest second, etc.
    # Since all values distinct, the smallest first value is 1.
    # Then from cell with value 1, we must move to a neighbor. Choose the neighbor with smallest value.
    # But wait - is greedy always optimal? Since we want lexicographically smallest sequence,
    # and at each step the choice only affects future options... 
    # Actually greedy works here because: the first element should be as small as possible.
    # The smallest possible first element is 1 (since 1 is in grid). So start at 1.
    # Then second element: must be a neighbor of 1. Choose smallest neighbor value.
    # But could choosing a larger second element lead to smaller third element that makes
    # overall sequence smaller? No - lexicographic comparison: smaller at earlier position wins.
    # So greedy is correct: at each step pick smallest reachable value.
    # But "reachable" - we can revisit cells, so from any cell we can go to any neighbor.
    # So at each step, from current cell, pick neighbor with min value.
    # However, we might also consider: could starting at a different cell give same first value? No, values distinct.
    # So start at value 1.
    
    # Wait, but is it always optimal to start at 1? Yes, because 1 is smallest possible first element.
    # Then greedily pick smallest neighbor.
    
    # But careful: what if k=1? Then just [1].
    
    # Let's verify with example 1: grid=[[1,2,3],[4,5,6],[7,8,9]], k=3
    # Start at 1 (0,0). Neighbors: 2 (0,1), 4 (1,0). Min is 2. Go to 2 (0,1).
    # From 2 (0,1): neighbors: 1 (0,0), 3 (0,2), 5 (1,1). Min is 1. Go to 1.
    # Path: [1,2,1]. Matches.
    
    # Example 2: k=1 -> [1]. Matches.
    
    # But is greedy always correct? Consider: from current cell, we pick smallest neighbor.
    # Could picking a larger neighbor now allow a much smaller next value?
    # Lexicographic: position i matters more than position i+1. So no.
    # But wait: what if picking smallest neighbor leads to dead end? No dead ends since we can revisit.
    # Actually we can always move (n>=2 so each cell has at least 2 neighbors? Not necessarily corner has 2).
    # But we can always move to some neighbor. So no dead end.
    
    # However, there's a subtlety: the greedy choice at step t depends only on current cell.
    # But could there be a tie? No, values distinct.
    # So greedy is optimal.
    
    # But wait - is it possible that starting at 1 is not optimal? 
    # Lexicographically, first element smallest possible is 1. Since 1 is in grid, yes start at 1.
    
    # So algorithm:
    # current = pos[1]
    # result = [1]
    # for _ in range(k-1):
    #     find neighbor of current with min value
    #     append that value, update current
    # return result
    
    # But wait - is it always true that from 1 we can reach the global minimum sequence?
    # Consider grid = [[1,100],[2,3]], k=3.
    # Start at 1 (0,0). Neighbors: 100 (0,1), 2 (1,0). Min is 2. Go to 2 (1,0).
    # From 2 (1,0): neighbors: 1 (0,0), 3 (1,1). Min is 1. Go to 1.
    # Path: [1,2,1].
    # Alternative: start at 1, go to 100? No, 2<100.
    # What if start at 2? First element 2 > 1, so worse.
    # So [1,2,1] is correct.
    
    # Another test: grid = [[1,5],[2,3]], k=4.
    # Start 1 (0,0). Neighbors: 5,2. Min 2. -> [1,2]
    # From 2 (1,0): neighbors: 1,3. Min 1. -> [1,2,1]
    # From 1 (0,0): neighbors: 5,2. Min 2. -> [1,2,1,2]
    # Is there better? Start 1, go to 2, go to 3? From 2, neighbors are 1 and 3. 1<3 so go to 1.
    # So [1,2,1,2] seems right.
    
    # But wait - what if from 1 we could go to 2, then from 2 go to 3, then from 3 go to ...?
    # But 1<3 so greedy picks 1. Lexicographically [1,2,1,...] < [1,2,3,...]. Correct.
    
    # So greedy works.
    
    # However, there's a catch: the problem says "It is guaranteed that the answer is unique."
    # This suggests greedy might not always be trivially correct? Let's think more.
    # Actually, is it possible that from 1, the smallest neighbor is X, but choosing X leads to
    # a sequence that is lexicographically larger than choosing a larger neighbor Y?
    # No, because at position 2, X < Y, so [1,X,...] < [1,Y,...] regardless of later elements.
    # So greedy is correct.
    
    # But wait - what if k is large and we need to consider that we might want to avoid
    # getting stuck in a cycle of small values? But lexicographically, smaller is always better
    # at the earliest differing position. So greedy is correct.
    
    # Actually, there's a known problem: "Minimum path of length k" where you can revisit.
    # The greedy approach: start at 1, then always go to smallest neighbor.
    # But is it possible that starting at 1 is not optimal? No, 1 is smallest.
    # But what if 1 is not reachable in a way that allows k steps? We can always move, so yes.
    
    # Wait, but what if from 1, the smallest neighbor is 2, but from 2, the smallest neighbor is 1,
    # so we oscillate 1,2,1,2,... Is that always optimal? 
    # Consider grid = [[1,2],[3,4]], k=3.
    # Start 1 (0,0). Neighbors: 2 (0,1), 3 (1,0). Min 2. -> [1,2]
    # From 2 (0,1): neighbors: 1 (0,0), 4 (1,1). Min 1. -> [1,2,1]
    # Alternative: 1->3->1? [1,3,1] > [1,2,1]. So yes.
    
    # But what if grid = [[1,4],[2,3]], k=3.
    # Start 1 (0,0). Neighbors: 4 (0,1), 2 (1,0). Min 2. -> [1,2]
    # From 2 (1,0): neighbors: 1 (0,0), 3 (1,1). Min 1. -> [1,2,1]
    # Alternative: 1->4->...? [1,4,...] > [1,2,...]. So yes.
    
    # So greedy seems correct.
    
    # But wait - is it possible that the smallest neighbor from 1 is not the best because
    # from that neighbor, the next smallest is large, while from a larger neighbor, the next
    # smallest is small? Lexicographically, position 2 matters more than position 3.
    # So [1,2,100] < [1,3,1] because 2<3 at position 2. So greedy is correct.
    
    # Therefore, the algorithm is:
    # 1. Find position of 1.
    # 2. result = [1]
    # 3. current = pos[1]
    # 4. For i in range(k-1):
    #       find neighbor with min value
    #       append that value
    #       current = that neighbor's position
    # 5. Return result
    
    # But wait - is it always optimal to start at 1? What if k=1? Then [1] is correct.
    # What if k>1? Starting at 1 gives smallest first element. Yes.
    
    # However, there's a known twist: sometimes the greedy approach fails because
    # you might need to look ahead. But here, since we can revisit, and values are distinct,
    # greedy is correct. Let me double-check with a potential counterexample.
    
    # Suppose grid = [[1,10],[2,3]], k=3.
    # Greedy: 1->2->1 => [1,2,1].
    # Alternative: 1->10->...? [1,10,...] worse.
    # Alternative: 2->1->2 => [2,1,2] worse (first element 2>1).
    # So [1,2,1] is min.
    
    # Suppose grid = [[1,3],[4,2]], k=3.
    # Greedy: 1 (0,0). Neighbors: 3 (0,1), 4 (1,0). Min 3. -> [1,3]
    # From 3 (0,1): neighbors: 1 (0,0), 2 (1,1). Min 1. -> [1,3,1]
    # Alternative: 1->4->1 => [1,4,1] worse.
    # Alternative: 1->3->2 => [1,3,2] > [1,3,1] since 1<2 at pos 3.
    # So [1,3,1] is min.
    
    # Suppose grid = [[1,5],[3,2]], k=3.
    # Greedy: 1 (0,0). Neighbors: 5 (0,1), 3 (1,0). Min 3. -> [1,3]
    # From 3 (1,0): neighbors: 1 (0,0), 2 (1,1). Min 1. -> [1,3,1]
    # Alternative: 1->5->...? [1,5,...] worse.
    # So [1,3,1] is min.
    
    # What if from 1, the smallest neighbor is 2, but from 2, the smallest neighbor is 3,
    # while from 1, the second smallest neighbor is 4, and from 4, the smallest neighbor is 2?
    # Then greedy: 1->2->3 => [1,2,3]
    # Alternative: 1->4->2 => [1,4,2] worse at pos 2.
    # So greedy still wins.
    
    # So greedy is correct.
    
    # But wait - is it possible that starting at 1 is not optimal because 1 is not the smallest
    # first element? No, 1 is the smallest value in grid.
    
    # So the algorithm is correct.
    
    # However, I recall a similar problem where greedy fails because you might need to
    # consider that from the smallest neighbor, you might be forced to go to a large value,
    # while from a slightly larger neighbor, you can go to a very small value.
    # But lexicographically, the smaller value at position 2 wins regardless of position 3.
    # So [1,2,100] < [1,3,1] because 2<3 at position 2.
    # So greedy is correct.
    
    # Therefore, implement greedy.
    
    # But wait - the problem says "It is guaranteed that the answer is unique."
    # This might imply that greedy might not always be unique? Or that there's only one
    # minimum path. Greedy gives one path. If greedy is correct, it's unique.
    
    # Let's implement.
    
    # Actually, I realize there might be a subtlety: what if from 1, the smallest neighbor is 2,
    # but from 2, the smallest neighbor is 1, so we get 1,2,1,2,... 
    # But what if k is large and we could get 1,2,1,2,... which is lexicographically smaller
    # than any other? Yes, that's the minimum.
    
    # But what if from 1, the smallest neighbor is 2, but from 2, the smallest neighbor is 3,
    # and from 3, the smallest neighbor is 1? Then 1,2,3,1,2,3,... 
    # Could there be a better path? 1,2,1,2,... would be better but not possible if 2's
    # smallest neighbor is 3. So 1,2,3,1,... is the greedy result.
    # Is there a better path? 1,3,...? [1,3,...] > [1,2,...]. So no.
    
    # So greedy is correct.
    
    # Let's code it.
    
    n = len(grid)
    pos = {}
    for i in range(n):
        for j in range(n):
            pos[grid[i][j]] = (i, j)
    
    # Start at 1
    cur_val = 1
    cur_pos = pos[1]
    result = [1]
    
    for _ in range(k - 1):
        i, j = cur_pos
        best_val = float('inf')
        best_pos = None
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n:
                if grid[ni][nj] < best_val:
                    best_val = grid[ni][nj]
                    best_pos = (ni, nj)
        result.append(best_val)
        cur_pos = best_pos
    
    return result


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
