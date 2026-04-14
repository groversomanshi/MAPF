import sys
sys.path.insert(0, '../')

import argparse
import yaml
from mic.a_star import AStar


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
        self.reservation_table = {}

    def get_neighbors(self, state):
        neighbors = []
        moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
        for dx, dy in moves:
            new_loc = Location(state.location.x + dx, state.location.y + dy)
            new_state = State(state.time + 1, new_loc)
            if self.state_valid(new_state) and self.not_in_reservation(new_state):
                neighbors.append(new_state)
        return neighbors

    def state_valid(self, state):
        return (0 <= state.location.x < self.dimension[0] and
                0 <= state.location.y < self.dimension[1] and
                (state.location.x, state.location.y) not in self.obstacles)

    def not_in_reservation(self, state):
        loc = (state.location.x, state.location.y)
        return (state.time, loc) not in self.reservation_table

    def admissible_heuristic(self, state, agent):
        goal = self.agent_dict[agent]["goal"]
        return abs(state.location.x - goal.location.x) + \
               abs(state.location.y - goal.location.y)

    def is_at_goal(self, state, agent):
        goal = self.agent_dict[agent]["goal"]
        return state.is_equal_except_time(goal)

    def make_agent_dict(self):
        for agent in self.agents:
            self.agent_dict[agent['name']] = {
                'start': State(0, Location(*agent['start'])),
                'goal': State(0, Location(*agent['goal']))
            }

    def commit_path(self, path):
        max_t = path[-1].time
        for state in path:
            key = (state.time, (state.location.x, state.location.y))
            self.reservation_table[key] = True
        for t in range(max_t + 1, max_t + 1000):
            key = (t, (path[-1].location.x, path[-1].location.y))
            self.reservation_table[key] = True

    def compute_solution_cost(self, solution):
        return sum(len(p) for p in solution.values())


class MiC(object):
    def __init__(self, env):
        self.env = env

    def search(self):
        solution = {}

        for agent in self.env.agent_dict:
            a_star = AStar(self.env)
            path = a_star.search(agent)

            if not path:
                return {}

            self.env.commit_path(path)
            solution[agent] = path

        print("solution found")
        return solution

    def build_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
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
    solution = solver.search()

    if not solution:
        print("Solution not found")
        return

    output = {
        "schedule": solver.build_plan(solution),
        "cost": env.compute_solution_cost(solution)
    }

    with open(args.output, 'w') as f:
        yaml.safe_dump(output, f)


if __name__ == "__main__":
    main()