import heapq

def a_star(graph, heuristics, start, goal):
    # Priority queue: (f = g + h, g, node, path)
    queue = [(heuristics[start], 0, start, [start])]
    visited = set()

    while queue:
        f, g, node, path = heapq.heappop(queue)

        if node in visited:
            continue
        visited.add(node)

        # Goal test
        if node == goal:
            return g, path

        # Explore neighbors
        for neighbor, cost in graph.get(node, []):
            if neighbor not in visited:
                g_new = g + cost
                f_new = g_new + heuristics.get(neighbor, 0)
                heapq.heappush(queue, (f_new, g_new, neighbor, path + [neighbor]))

    return float('inf'), []  # If goal not reachable

# Example weighted graph
graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 3), ('E', 1)],
    'C': [('F', 5)],
    'D': [('G', 3)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

# Heuristic values (estimated distance to goal)
heuristics = {
    'A': 7,
    'B': 6,
    'C': 5,
    'D': 3,
    'E': 2,
    'F': 3,
    'G': 0
}

start_node = 'A'
goal_node = 'G'

cost, path = a_star(graph, heuristics, start_node, goal_node)

print("A* Search Result:")
print("Path:", " -> ".join(path))
print("Total Cost:", cost)
