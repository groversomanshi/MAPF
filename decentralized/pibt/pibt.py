import argparse
import yaml
import heapq

from utils.a_star import Location

class Environment:
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = set(map(tuple, obstacles))
        self.agents = agents
        self.agent_dict = {}
        self.distance_maps = {}
        self.make_agent_dict()
        self.precompute_distance_maps()

    def make_agent_dict(self):
        for agent in self.agents:
            name = agent['name']
            self.agent_dict[name] = {
                'start': Location(*agent['start']),
                'goal': Location(*agent['goal'])
            }

    def in_bounds(self, loc):
        return 0 <= loc.x < self.dimension[0] and 0 <= loc.y < self.dimension[1]

    def is_obstacle(self, loc):
        return (loc.x, loc.y) in self.obstacles

    def get_neighbors(self, loc):
        neighbors = []
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_loc = Location(loc.x + dx, loc.y + dy)
            if self.in_bounds(new_loc) and not self.is_obstacle(new_loc):
                neighbors.append(new_loc)
        return neighbors
    
    def precompute_distance_maps(self):
        for agent, data in self.agent_dict.items():
            self.distance_maps[agent] = self.compute_distance_map(data['goal'])

    def compute_distance_map(self, goal):
        distances = {(goal.x, goal.y): 0}
        heap = [(0, goal.x, goal.y)]
        while heap:
            dist, x, y = heapq.heappop(heap)
            loc = Location(x, y)
            if dist > distances[(x, y)]:
                continue

            for nbh in self.get_neighbors(loc):
                key = (nbh.x, nbh.y)
                new_dist = dist + 1
                if new_dist < distances.get(key, float('inf')):
                    distances[key] = new_dist
                    heapq.heappush(heap, (new_dist, nbh.x, nbh.y))
        return distances

    def distance_to_goal(self, loc, agent):
        return self.distance_maps[agent].get((loc.x, loc.y), float('inf'))

    def compute_solution_cost(self, solution):
        return sum(path[-1]['t'] for path in solution.values())

class PIBT:
    def __init__(self, env):
        self.env = env
        self.priorities = {agent: 0 for agent in self.env.agent_dict}
        self.positions = {}
        self.next_positions = {}
        self.undecided = set()
        self.occupied = set()

    def pibt(self, a_i, a_j=None):
        self.undecided.remove(a_i)
        candidates = self.candidate_locations(a_i, a_j)

        for loc in candidates:
            self.occupied.add(loc)
            blocker = None

            for other in self.undecided:
                if self.positions[other] == loc:
                    blocker = other
                    break

            if blocker is not None:
                if self.pibt(blocker, a_i):
                    self.next_positions[a_i] = loc
                    return True
            else:
                self.next_positions[a_i] = loc
                return True

            self.occupied.remove(loc)

        self.next_positions[a_i] = self.positions[a_i]
        return False

    def update_priorities(self):
        for agent in self.env.agent_dict:
            if self.positions[agent] == self.env.agent_dict[agent]['goal']:
                self.priorities[agent] = 0
            else:
                self.priorities[agent] += 1
    
    def all_at_goals(self):
        for agent in self.env.agent_dict:
            if self.positions[agent] != self.env.agent_dict[agent]['goal']:
                return False
        return True

    def to_yaml(self, paths):
        lines = {}
        for agent, path in paths.items():
            lines[agent] = [{'t': t, 'x': loc.x, 'y': loc.y} for t, loc in enumerate(path)]
        return lines

    def search(self, max_timesteps=None):
        if max_timesteps is None:
            width, height = self.env.dimension
            num_agents = len(self.env.agent_dict)
            diam = (width - 1) + (height - 1)
            max_timesteps = diam * num_agents

        self.positions = {agent: data['start'] for agent, data in self.env.agent_dict.items()}
        paths = {agent: [loc] for agent, loc in self.positions.items()}

        for _ in range(max_timesteps):
            if self.all_at_goals():
                print("Solution found")
                return self.to_yaml(paths)

            self.update_priorities()
            self.next_positions = {}
            self.undecided = set(self.env.agent_dict.keys())
            self.occupied = set()

            while self.undecided:
                agent = max(self.undecided, key=lambda a: self.priorities[a])
                self.pibt(agent)

            self.positions = dict(self.next_positions)
            for agent, loc in self.positions.items():
                paths[agent].append(loc)

        return {}

    def candidate_locations(self, a_i, a_j=None):
        current = self.positions[a_i]
        candidates = []

        for loc in self.env.get_neighbors(current):
            if loc in self.occupied:
                continue
            if a_j is not None and loc == self.positions[a_j]:
                continue
            candidates.append(loc)
        candidates.sort(key=lambda loc: self.env.distance_to_goal(loc, a_i))
        return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    args = parser.parse_args()

    # Read from input file
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']

    env = Environment(dimension, agents, obstacles)

    # # Searching
    pibt = PIBT(env)
    solution = pibt.search(param.get("max_timesteps"))
    if not solution:
        print("Solution not found")
        return

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()