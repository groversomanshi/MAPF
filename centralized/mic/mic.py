import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.a_star import SingleAgentAStar, Location as UtilsLoc

import argparse
import yaml
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

    def get_neighbors(self, state):
        neighbors = []
        moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
        for dx, dy in moves:
            new_loc = Location(state.location.x + dx, state.location.y + dy)
            new_state = State(state.time + 1, new_loc)
            if self.state_valid(new_state):
                neighbors.append(new_state)
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


class MiC(object):
    """
    MRP-IC: Multi-Robot Motion Planning by Incremental Coordination.
    Saha & Isto, IROS 2006.

    Robots are ordered and planned sequentially.  Each robot i is planned
    with a space-time A* that treats the already-committed paths of robots
    1 … i-1 as dynamic obstacles (the reservation table), effectively
    searching the composite space P_{i-1} × C_i without enumerating it.
    If planning fails for any robot the whole pipeline restarts with a
    new random ordering (not implemented here — a fixed ordering is used).
    """

    def __init__(self, env):
        self.env = env

    def _build_reservation_table(self, committed_paths):
        """
        Map (time, (x, y)) → set-of-agent-names for all time steps,
        including an infinite wait at the goal after the path ends.
        """
        table = {}
        for agent, path in committed_paths.items():
            max_t = path[-1].time
            for state in path:
                key = (state.time, (state.location.x, state.location.y))
                table.setdefault(key, set()).add(agent)
            # Hold at goal forever
            goal = path[-1]
            for t in range(max_t + 1, max_t + 1000):
                key = (t, (goal.location.x, goal.location.y))
                table.setdefault(key, set()).add(agent)
        return table

    def search(self):
        agents = list(self.env.agent_dict.keys())
        committed_paths = {}

        for agent in agents:
            reservation_table = self._build_reservation_table(committed_paths)
            a_star = SingleAgentAStar(self.env.dimension, self.env.obstacles)
            start = UtilsLoc(self.env.agent_dict[agent]["start"].location.x, self.env.agent_dict[agent]["start"].location.y)
            goal = UtilsLoc(self.env.agent_dict[agent]["goal"].location.x, self.env.agent_dict[agent]["goal"].location.y)

            path = a_star.search(agent, start, goal, reservation_table=reservation_table)

            if path is None:
                return {}

            converted_path = []
            for s in path:
                converted_path.append(State(s.time, Location(s.location.x, s.location.y)))

            committed_paths[agent] = converted_path

        print("solution found")
        return committed_paths

    def build_plan(self, paths):
        plan = {}
        for agent, path in paths.items():
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

    solver = MiC(env)

    start_time = time.perf_counter()
    paths = solver.search()
    elapsed = time.perf_counter() - start_time

    if not paths:
        print("Solution not found")
        return

    solution = solver.build_plan(paths)

    cost = env.compute_solution_cost(paths)
    makespan = max(path[-1]['t'] for path in solution.values())

    print(f"Computation time : {elapsed:.4f}s")
    print(f"Makespan         : {makespan}")
    print(f"Cost (sum of path lengths) : {cost}")

    output = {
        "schedule": solution,
        "cost": cost
    }

    with open(args.output, 'w') as f:
        yaml.safe_dump(output, f)


if __name__ == "__main__":
    main()