import math

# Minimax function with Alpha-Beta Pruning
def alphabeta(depth, nodeIndex, maximizingPlayer, values, alpha, beta):
    
    # If leaf node → return value
    if depth == 3:  
        return values[nodeIndex]

    if maximizingPlayer:
        best = -math.inf

        # Explore left child then right child
        for i in range(2):
            val = alphabeta(depth + 1, nodeIndex * 2 + i, False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)

            # Beta Cutoff
            if beta <= alpha:
                break

        return best

    else:
        best = math.inf

        # Explore left child then right child
        for i in range(2):
            val = alphabeta(depth + 1, nodeIndex * 2 + i, True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)

            # Alpha Cutoff
            if beta <= alpha:
                break

        return best


# Driver Code
if __name__ == "__main__":

    # Leaf node values (8 leaves → depth 3 binary tree)
    values = [3, 5, 6, 9, 1, 2, 0, -1]

    print("Leaf Nodes =", values)

    result = alphabeta(0, 0, True, values, -math.inf, math.inf)

    print("Optimal Value =", result)
