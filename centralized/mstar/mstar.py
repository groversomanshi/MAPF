import sys
sys.path.insert(0, '../')

import argparse
import yaml
import heapq
from itertools import product
from copy import deepcopy
from math import fabs


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
        self.states = {}
        self.paths = {}
        self.collisions = frozenset()
        self.cost = 0
        self.h = 0
        self.parent = None

    def __lt__(self, other):
        return (self.cost + self.h) < (other.cost + other.h)


class MStar:
    def __init__(self, env):
        self.env = env
        self.open_list = []
        # Map from node key → Node, so we can reopen nodes after backprop
        self.node_table = {}

    def search(self):
        start = Node()

        for agent in self.env.agent_dict:
            s = self.env.agent_dict[agent]["start"]
            start.states[agent] = s
            start.paths[agent] = [s]

        start.cost = 0
        start.h = self.heuristic(start)
        start.collisions = frozenset()

        heapq.heappush(self.open_list, start)
        key = self.hash_node(start)
        self.node_table[key] = start

        closed = set()

        while self.open_list:
            curr = heapq.heappop(self.open_list)

            if self.is_goal(curr):
                print("solution found")
                return self.build_plan(curr)

            key = self.hash_node(curr)
            if key in closed:
                continue
            closed.add(key)

            for nxt in self.expand(curr):
                nxt_key = self.hash_node(nxt)
                if nxt_key not in closed:
                    heapq.heappush(self.open_list, nxt)
                    self.node_table[nxt_key] = nxt

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

        # Coupled agents get full neighbor expansion; uncoupled follow optimal policy
        moves = {}
        for a in agents:
            if a in node.collisions:
                moves[a] = self.env.get_neighbors(node.states[a])
            else:
                moves[a] = self.env.optimal_next(node.states[a], a)

        combos = list(product(*[moves[a] for a in agents]))
        new_nodes = []

        for combo in combos:
            new_node = Node()
            new_node.parent = node
            new_node.states = dict(zip(agents, combo))

            new_node.paths = deepcopy(node.paths)
            for a in agents:
                new_node.paths[a] = node.paths[a] + [new_node.states[a]]

            collisions = self.detect_collision(node, new_node)

            if collisions:
                # Backpropagate new collisions up the search tree and reopen ancestors
                self.backprop(node, collisions)
                new_node.collisions = node.collisions | collisions
            else:
                new_node.collisions = node.collisions

            # Cost = sum of path lengths so far (sum-of-costs)
            new_node.cost = sum(len(new_node.paths[a]) for a in agents)
            new_node.h = self.heuristic(new_node)

            new_nodes.append(new_node)

        return new_nodes

    def backprop(self, node, new_collisions):
        """
        Propagate newly discovered colliding agents up through ancestors.
        Any ancestor whose collision set grows must be reopened so it is
        re-expanded with the larger coupled set.
        """
        if node is None:
            return

        added = new_collisions - node.collisions
        if not added:
            return  # Nothing new — stop propagating

        node.collisions = node.collisions | new_collisions
        # Reopen this node so it gets re-expanded with the updated collision set
        heapq.heappush(self.open_list, node)
        # Recurse upward
        self.backprop(node.parent, new_collisions)

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

        return frozenset(col)

    def hash_node(self, node):
        """
        A node is uniquely identified by the joint position of all agents
        AND the current collision set (which determines expansion behaviour).
        """
        return (
            tuple(
                (a, node.states[a].location.x, node.states[a].location.y)
                for a in sorted(node.states)
            ),
            node.collisions,
        )

    def build_plan(self, node):
        plan = {}
        for agent, path in node.paths.items():
            plan[agent] = [
                {'t': s.time, 'x': s.location.x, 'y': s.location.y}
                for s in path
            ]
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
    solution = solver.search()

    if not solution:
        print("Solution not found")
        return

    output = {
        "schedule": solution,
        "cost": env.compute_solution_cost(solution)
    }

    with open(args.output, 'w') as f:
        yaml.safe_dump(output, f)


if __name__ == "__main__":
    main()