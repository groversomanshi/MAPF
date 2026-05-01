import sys
sys.path.insert(0, '../')

import argparse
import yaml
import heapq
from itertools import product
from math import fabs
import time


class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return str((self.x, self.y))


class State(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash((self.time, self.location.x, self.location.y))

    def is_equal_except_time(self, state):
        return self.location == state.location


class Environment(object):
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = set(map(tuple, obstacles))
        self.agents = agents
        self.agent_dict = {}
        self.make_agent_dict()
        # Precompute optimal cost-to-go for each agent via reverse Dijkstra
        self._policy_cache = {}
        self._precompute_policies()

    def get_neighbors(self, state):
        neighbors = []
        moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
        for dx, dy in moves:
            new_loc = Location(state.location.x + dx, state.location.y + dy)
            new_state = State(state.time + 1, new_loc)
            if self.state_valid(new_state):
                neighbors.append(new_state)
        return neighbors

    def get_neighbors_loc(self, loc):
        """Return neighboring locations (ignoring time) — used for policy precomputation."""
        neighbors = []
        moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
        for dx, dy in moves:
            nx, ny = loc[0] + dx, loc[1] + dy
            if (0 <= nx < self.dimension[0] and
                    0 <= ny < self.dimension[1] and
                    (nx, ny) not in self.obstacles):
                neighbors.append((nx, ny))
        return neighbors

    def state_valid(self, state):
        return (0 <= state.location.x < self.dimension[0] and
                0 <= state.location.y < self.dimension[1] and
                (state.location.x, state.location.y) not in self.obstacles)

    def admissible_heuristic(self, state, agent):
        # Use precomputed exact cost-to-go if available, else fall back to Manhattan
        dist = self._policy_cache.get(agent, {})
        loc = (state.location.x, state.location.y)
        if loc in dist:
            return dist[loc]
        goal = self.agent_dict[agent]["goal"]
        return fabs(state.location.x - goal.location.x) + \
               fabs(state.location.y - goal.location.y)

    def is_at_goal(self, state, agent):
        goal = self.agent_dict[agent]["goal"]
        return state.is_equal_except_time(goal)

    def make_agent_dict(self):
        for agent in self.agents:
            self.agent_dict[agent['name']] = {
                'start': State(0, Location(*agent['start'])),
                'goal': State(0, Location(*agent['goal']))
            }

    def compute_solution_cost(self, solution):
        return sum(len(p) for p in solution.values())

    def _precompute_policies(self):
        """
        For each agent, run reverse Dijkstra from the goal to compute the
        exact cost-to-go (distance) from every reachable cell.
        This gives us the true optimal individual policy.
        """
        for agent_name, data in self.agent_dict.items():
            goal_loc = (data['goal'].location.x, data['goal'].location.y)
            dist = {}
            heap = [(0, goal_loc)]
            dist[goal_loc] = 0

            while heap:
                d, loc = heapq.heappop(heap)
                if d > dist.get(loc, float('inf')):
                    continue
                for nb in self.get_neighbors_loc(loc):
                    nd = d + 1
                    if nd < dist.get(nb, float('inf')):
                        dist[nb] = nd
                        heapq.heappush(heap, (nd, nb))

            self._policy_cache[agent_name] = dist

    def optimal_next(self, state, agent):
        """
        Return the single best neighbor for `agent` from `state` according
        to the precomputed cost-to-go table.  Returns a list so it plugs
        directly into the moves dict in expand().
        """
        dist = self._policy_cache[agent]
        neighbors = self.get_neighbors(state)
        if not neighbors:
            return [state]

        best = min(
            neighbors,
            key=lambda s: dist.get((s.location.x, s.location.y), float('inf'))
        )
        return [best]


class Node:
    def __init__(self):
        self.states = {}                # agent -> current State
        self.collisions = frozenset()   # agents involved in any collision at/through this node
        self.backprop_set = set()       # set of Node objects that have this node as a successor
        self.cost = 0                   # g: cost so far (sum-of-costs)
        self.h = 0                      # heuristic cost-to-go
        self.parent = None              # for path reconstruction

    def __lt__(self, other):
        return (self.cost + self.h) < (other.cost + other.h)


class MStar:
    def __init__(self, env):
        self.env = env
        self.open_list = []
        # Map from position-key -> Node (one node per joint position)
        self.node_table = {}
        self.num_conflicts = 0
        self.num_backprops = 0
        self.iterations = 0

    def _pos_key(self, node):
        """
        Unique key based solely on joint position (not collision set).
        Per the paper, each joint configuration maps to exactly one node;
        the collision set is mutable state on that node.
        """
        return tuple(
            (a, node.states[a].location.x, node.states[a].location.y)
            for a in sorted(node.states)
        )

    def _closed_key(self, node):
        """
        Key for the closed set that includes the collision set.
        This is critical: when backprop grows a node's collision set, that node
        must be re-expanded even if its position was seen before — because it
        will now generate different (more coupled) successors.
        """
        return (self._pos_key(node), node.collisions)

    def search(self):
        start = Node()
        for agent in self.env.agent_dict:
            start.states[agent] = self.env.agent_dict[agent]["start"]

        start.cost = 0
        start.h = self.heuristic(start)
        start.collisions = frozenset()

        key = self._pos_key(start)
        self.node_table[key] = start
        heapq.heappush(self.open_list, start)

        # Closed set keyed on (position, collision_set) so that nodes reopened
        # with a larger collision set are not skipped — they need re-expansion.
        closed = set()

        while self.open_list:
            curr = heapq.heappop(self.open_list)
            self.iterations += 1

            if self.iterations % 1000 == 0:
                print(f"iter {self.iterations}, open list size: {len(self.open_list)}, backprops: {self.num_backprops}")

            if self.is_goal(curr):
                print("solution found")
                return self.build_plan(curr)

            closed_key = self._closed_key(curr)
            if closed_key in closed:
                continue
            closed.add(closed_key)

            for nxt in self.expand(curr):
                nxt_closed_key = self._closed_key(nxt)
                if nxt_closed_key not in closed:
                    # Add curr to nxt's backpropagation set (curr considered nxt as successor)
                    nxt.backprop_set.add(curr)
                    heapq.heappush(self.open_list, nxt)

        return {}

    def is_goal(self, node):
        return all(
            self.env.is_at_goal(node.states[a], a)
            for a in node.states
        )

    def heuristic(self, node):
        return sum(
            self.env.admissible_heuristic(node.states[a], a)
            for a in node.states
        )

    def expand(self, node):
        agents = list(self.env.agent_dict.keys())

        # Coupled agents (in collision set) get full neighbor expansion;
        # uncoupled agents follow their individually optimal policy.
        moves = {}
        for a in agents:
            if a in node.collisions:
                moves[a] = self.env.get_neighbors(node.states[a])
            else:
                moves[a] = self.env.optimal_next(node.states[a], a)

        combos = list(product(*[moves[a] for a in agents]))
        new_nodes = []

        for combo in combos:
            new_states = dict(zip(agents, combo))

            nxt_key = tuple(
                (a, new_states[a].location.x, new_states[a].location.y)
                for a in sorted(new_states)
            )

            # Reuse existing node if we've seen this position before (single node per position)
            if nxt_key in self.node_table:
                nxt = self.node_table[nxt_key]
                new_cost = node.cost + len(agents)
                # Update if we found a cheaper path to this position
                if new_cost < nxt.cost:
                    nxt.cost = new_cost
                    nxt.parent = node
                    heapq.heappush(self.open_list, nxt)
            else:
                nxt = Node()
                nxt.states = new_states
                nxt.cost = node.cost + len(agents)  # each agent takes one step
                nxt.h = self.heuristic(nxt)
                nxt.parent = node
                self.node_table[nxt_key] = nxt

            # Detect collisions in this transition; if found, backpropagate and skip
            # this successor — it will be reconsidered once the collision set is updated
            collisions = self.detect_collision(node, nxt)
            if collisions:
                self.backprop(node, collisions)
                continue

            new_nodes.append(nxt)

        return new_nodes

    def backprop(self, node, new_collisions, visited=None):
        """
        Propagate newly discovered colliding agents through the backpropagation
        set (all predecessors that have explored paths through this node), not just
        the single tree parent.  This matches Section 3 of the paper.
        """
        if visited is None:
            visited = set()

        if node is None or node in visited:
            return

        visited.add(node)

        added = new_collisions - node.collisions
        if not added:
            return  # Nothing new to propagate — stop

        self.num_backprops += 1
        node.collisions |= new_collisions
        # Reopen this node so it is re-expanded with the updated (larger) collision set.
        # Because the closed set includes the collision set, this reopened node will
        # not be skipped — its new (pos, collisions) key won't be in closed yet.
        heapq.heappush(self.open_list, node)
        # Recurse to all predecessors in the backpropagation set
        for pred in node.backprop_set:
            self.backprop(pred, new_collisions, visited)

    def detect_collision(self, prev, curr):
        agents = list(curr.states.keys())
        col = set()

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a1, a2 = agents[i], agents[j]

                p1, p2 = prev.states[a1], prev.states[a2]
                c1, c2 = curr.states[a1], curr.states[a2]

                # Vertex collision
                if c1.location == c2.location:
                    col |= {a1, a2}

                # Edge (swap) collision
                if (p1.location == c2.location and
                        p2.location == c1.location):
                    col |= {a1, a2}

        if col:
            self.num_conflicts += 1

        return frozenset(col)

    def build_plan(self, goal_node):
        """
        Reconstruct paths by walking up the parent chain from the goal node.
        """
        # Walk up the parent chain to collect the sequence of joint states
        chain = []
        node = goal_node
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()

        plan = {agent: [] for agent in self.env.agent_dict}
        for t, n in enumerate(chain):
            for agent in self.env.agent_dict:
                state = n.states[agent]
                plan[agent].append({'t': t, 'x': state.location.x, 'y': state.location.y})

        return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param")
    parser.add_argument("output")
    args = parser.parse_args()

    with open(args.param, 'r') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]

    env = Environment(dimension, agents, obstacles)

    solver = MStar(env)

    start_time = time.perf_counter()
    solution = solver.search()
    elapsed = time.perf_counter() - start_time

    if not solution:
        print("Solution not found")
        return

    cost = env.compute_solution_cost(solution)
    makespan = max(path[-1]['t'] for path in solution.values())

    print(f"Computation time : {elapsed:.4f}s")
    print(f"Makespan         : {makespan}")
    print(f"Cost (sum of path lengths) : {cost}")
    print(f"Number of conflicts detected : {solver.num_conflicts}")

    output = {
        "schedule": solution,
        "cost": cost
    }

    with open(args.output, 'w') as f:
        yaml.safe_dump(output, f)


if __name__ == "__main__":
    main()