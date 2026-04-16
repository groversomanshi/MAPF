import heapq
import itertools
from collections import deque

class MStarNode:
    def __init__(self, positions):
        self.positions = tuple(positions)
        self.heuristic = 0
        self.collision_set = set() # Set of agents (indices) in collision
        self.back_set = set()      # Predecessors that generated this state
        self.back_ptr = None       # Pointer for the best path to root
        self.cost_to_come = float('inf')

    def __lt__(self, other):
        return (self.cost_to_come + self.heuristic) < (other.cost_to_come + other.heuristic)
    
    def __eq__(self, other):
        return self.positions == other.positions
    
    def __hash__(self):
        return hash(self.positions)

def get_neighbors(pos, grid):
    moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
    neighbors = []
    for d in moves:
        nx, ny = pos[0] + d[0], pos[1] + d[1]
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]):
            if grid[ny][nx] == 0:  # 0 indicates free space
                neighbors.append((nx, ny))
    return neighbors

def compute_policy(goal, grid):
    # Backward BFS from goal to compute shortest paths for a single agent
    # Returns a dist map and a policy map
    queue = deque([goal])
    dist = {goal: 0}
    policy = {goal: [(0,0)]} # best moves to reach goal (0,0) implies waiting at goal
    
    while queue:
        curr = queue.popleft()
        for nx, ny in get_neighbors(curr, grid):
            dx, dy = curr[0] - nx, curr[1] - ny
            if (nx, ny) not in dist:
                dist[(nx, ny)] = dist[curr] + 1
                policy[(nx, ny)] = [(dx, dy)]
                queue.append((nx, ny))
            elif dist[(nx, ny)] == dist[curr] + 1:
                policy[(nx, ny)].append((dx, dy))
    return dist, policy

def check_collisions(positions, prev_positions=None):
    # Returns a set of agent indices that are colliding
    colliding = set()
    n = len(positions)
    # Vertex collisions
    pos_map = {}
    for i, p in enumerate(positions):
        if p in pos_map:
            colliding.add(i)
            colliding.add(pos_map[p])
        else:
            pos_map[p] = i
    
    # Edge collisions (swapping places)
    if prev_positions:
        for i in range(n):
            for j in range(i+1, n):
                if positions[i] == prev_positions[j] and positions[j] == prev_positions[i]:
                    colliding.add(i)
                    colliding.add(j)
    return colliding

def mstar(grid, starts, goals):
    num_agents = len(starts)
    
    # Precompute individual policies and heuristics
    policies = []
    dists = []
    for i in range(num_agents):
        d, p = compute_policy(goals[i], grid)
        dists.append(d)
        policies.append(p)
        
    start_node = MStarNode(starts)
    start_node.cost_to_come = 0
    start_node.heuristic = sum(dists[i][starts[i]] for i in range(num_agents))
    
    open_list = []
    heapq.heappush(open_list, start_node)
    
    graph = {start_node.positions: start_node} # Visited nodes
    
    def backprop(vk, cv):
        # cv is the collision set from vk
        # If vk's collision set changes, we might need to re-evaluate it
        vk_node = graph[vk]
        is_subset = cv.issubset(vk_node.collision_set)
        if not is_subset:
            vk_node.collision_set.update(cv)
            if vk_node not in open_list:
                heapq.heappush(open_list, vk_node)
            for vl in vk_node.back_set:
                backprop(vl, vk_node.collision_set)

    while open_list:
        vk = heapq.heappop(open_list)
        
        # Check if goal reached
        if vk.positions == tuple(goals):
            # Reconstruct path
            path = []
            curr = vk
            while curr:
                path.append(curr.positions)
                curr = curr.back_ptr
            return path[::-1] # Reverse
            
        # Determine allowed actions for each agent
        actions = []
        for i in range(num_agents):
            pos = vk.positions[i]
            if i in vk.collision_set:
                # Agent is in a collision set, expand to all neighbors
                valid_actions = get_neighbors(pos, grid)
                actions.append(valid_actions)
            else:
                # Agent is not in collision set, follow optimal policy
                if pos == goals[i]:
                    actions.append([pos])
                else:
                    best_moves = policies[i][pos]
                    # Select the first best move
                    m = best_moves[0]
                    actions.append([(pos[0]+m[0], pos[1]+m[1])])
                    
        # Filter combinations where the same agent is moving
        for next_pos in itertools.product(*actions):
            next_pos_tuple = tuple(next_pos)
            
            if next_pos_tuple not in graph:
                vl = MStarNode(next_pos_tuple)
                vl.heuristic = sum(dists[i].get(next_pos_tuple[i], float('inf')) for i in range(num_agents))
                graph[next_pos_tuple] = vl
            else:
                vl = graph[next_pos_tuple]
                
            vl.back_set.add(vk.positions)
            
            # Get collisions
            colls = check_collisions(next_pos_tuple, vk.positions)
            if colls:
                vl.collision_set.update(colls)
                backprop(vk.positions, vl.collision_set)
                
            # Update path if optimal. In M*, we only consider transitions valid if NO collisions occur 
            # (or rather, the collision set logic forces them to evade it eventually)
            if vk.cost_to_come + 1 < vl.cost_to_come and len(colls) == 0:
                vl.cost_to_come = vk.cost_to_come + 1
                vl.back_ptr = vk
                heapq.heappush(open_list, vl)

    return None # No path found

if __name__ == '__main__':
    # Simple grid world: 0 is free, 1 is obstacle
    # 5x5 grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    # Agent 0 goes from top-left (0,0) to bottom-right (4,4)
    # Agent 1 goes from top-right (4,0) to bottom-left (0,4)
    starts = [(0, 0), (4, 0)]
    goals = [(4, 4), (0, 4)]
    
    print("Running Basic M*...")
    path = mstar(grid, starts, goals)
    if path:
        print("Path found!")
        for step, positions in enumerate(path):
            print(f"Time {step}: {positions}")
    else:
        print("No path found.")
