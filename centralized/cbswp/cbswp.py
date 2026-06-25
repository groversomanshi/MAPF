import argparse
import yaml
import heapq
import itertools

from utils.a_star import Location, SingleAgentAStar

class Environment:
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = set(map(tuple, obstacles))
        self.agents = agents
        self.agent_dict = {}
        self.planner = SingleAgentAStar(dimension, obstacles)
        self.make_agent_dict()

    def make_agent_dict(self):
        for agent in self.agents:
            name = agent["name"]
            self.agent_dict[name] = {
                "start": Location(*agent["start"]),
                "goal": Location(*agent["goal"])
            }

    def plan_for_agent(self, agent, reservation_table=None):
        return self.planner.search(agent, self.agent_dict[agent]["start"], self.agent_dict[agent]["goal"], reservation_table=reservation_table)

    @staticmethod
    def compute_solution_cost(solution):
        return sum(path[-1].time for path in solution.values())

    @staticmethod
    def compute_yaml_cost(solution):
        return sum(len(path) for path in solution.values())

    @staticmethod
    def get_state(agent, solution, time):
        path = solution[agent]
        if time < len(path):
            return path[time]
        return path[-1]
    
    def get_conflicts(self, solution):
        conflicts = []
        agents = sorted(solution.keys())
        max_t = max(len(path) for path in solution.values())

        for t in range(max_t):
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    agent_1 = agents[i]
                    agent_2 = agents[j]
                    state_1 = self.get_state(agent_1, solution, t)
                    state_2 = self.get_state(agent_2, solution, t)

                    if state_1.location == state_2.location:
                        conflicts.append(Conflict(t, Conflict.VERTEX, agent_1, agent_2))

            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    agent_1 = agents[i]
                    agent_2 = agents[j]
                    state_1a = self.get_state(agent_1, solution, t)
                    state_1b = self.get_state(agent_1, solution, t + 1)
                    state_2a = self.get_state(agent_2, solution, t)
                    state_2b = self.get_state(agent_2, solution, t + 1)

                    if state_1a.location == state_2b.location and state_1b.location == state_2a.location:
                        conflicts.append(Conflict(t, Conflict.EDGE, agent_1, agent_2))

        return conflicts
    
    @staticmethod
    def higher_priority_agents(agent, priority_constraints):
        higher_agents = set()
        changed = True

        while changed:
            changed = False
            for constraint in priority_constraints:
                if constraint.lower == agent or constraint.lower in higher_agents:
                    if constraint.higher not in higher_agents:
                        higher_agents.add(constraint.higher)
                        changed = True
        return higher_agents

class Conflict:
    VERTEX = "vertex"
    EDGE = "edge"

    def __init__(self, time, conflict_type, agent_1, agent_2):
        self.time = time
        self.type = conflict_type
        self.agent_1 = agent_1
        self.agent_2 = agent_2
    
class PriorityConstraint:
    def __init__(self, higher, lower):
        self.higher = higher
        self.lower = lower

    def __eq__(self, other):
        return self.higher == other.higher and self.lower == other.lower

    def __hash__(self):
        return hash((self.higher, self.lower))

class HighLevelNode:
    def __init__(self):
        self.solution = {}
        self.priority_constraints = set()
        self.cost = 0

    def copy(self):
        node = HighLevelNode()
        node.solution = {agent: list(path) for agent, path in self.solution.items()}
        node.priority_constraints = set(self.priority_constraints)
        node.cost = self.cost
        return node

class CBSWP:
    def __init__(self, env):
        self.env = env
        self.open_list = []
        self.counter = itertools.count()

    def compute_solution(self, priority_constraints):
        solution = {}
        for agent in self.env.agent_dict:
            reservation_table = self.build_reservation_table(agent, priority_constraints, solution)
            path = self.env.plan_for_agent(agent, reservation_table)
            if path is None:
                return None
            solution[agent] = path
        return solution

    def push(self, node):
        heapq.heappush(self.open_list, (node.cost, next(self.counter), node))

    def build_reservation_table(self, agent, priority_constraints, solution):
        reservation_table = {}
        higher_agents = self.env.higher_priority_agents(agent, priority_constraints)

        for higher_agent in higher_agents:
            if higher_agent not in solution:
                continue
            path = solution[higher_agent]

            for state in path:
                key = (state.time, (state.location.x, state.location.y))
                reservation_table.setdefault(key, []).append(higher_agent)

            goal = path[-1]
            for t in range(goal.time + 1, goal.time + 1000):
                key = (t, (goal.location.x, goal.location.y))
                reservation_table.setdefault(key, []).append(higher_agent)
        return reservation_table

    @staticmethod
    def creates_cycle(priority_constraints):
        graph = {}
        for constraint in priority_constraints:
            graph.setdefault(constraint.higher, set()).add(constraint.lower)
        
        visiting = set()
        visited = set()

        def visit(agent):
            if agent in visiting:
                return True
            if agent in visited:
                return False
            
            visiting.add(agent)
            for lower in graph.get(agent, set()):
                if visit(lower):
                    return True
            visiting.remove(agent)
            visited.add(agent)
            return False
        
        return any(visit(agent) for agent in graph)

    def choose_conflict(self, node):
        conflicts = self.env.get_conflicts(node.solution)
        if not conflicts:
            return None
        return min(conflicts, key=lambda conflict: conflict.time)

    def branch_on_conflict(self, node, higher, lower):
        child = node.copy()
        child.priority_constraints.add(PriorityConstraint(higher, lower))
        if self.creates_cycle(child.priority_constraints):
            return None

        reservation_table = self.build_reservation_table(lower, child.priority_constraints, child.solution)
        new_path = self.env.plan_for_agent(lower, reservation_table)
        if new_path is None:
            return None

        child.solution[lower] = new_path
        child.cost = self.env.compute_solution_cost(child.solution)
        return child

    def search(self):
        self.open_list = []
        self.counter = itertools.count()

        root = HighLevelNode()
        root.solution = self.compute_solution(root.priority_constraints)
        if root.solution is None:
            return {}
        root.cost = self.env.compute_solution_cost(root.solution)
        self.push(root)

        closed = set()

        while self.open_list:
            _, _, node = heapq.heappop(self.open_list)
            signature = self.node_signature(node)
            if signature in closed:
                continue
            closed.add(signature)

            conflict = self.choose_conflict(node)
            if conflict is None:
                print("Solution found")
                return self.to_yaml(node.solution)

            for higher, lower in ((conflict.agent_1, conflict.agent_2), (conflict.agent_2, conflict.agent_1)):
                child = self.branch_on_conflict(node, higher, lower)
                if child is not None:
                    self.push(child)

        return {}

    @staticmethod
    def to_yaml(solution):
        output = {}
        for agent, path in solution.items():
            output[agent] = [{"t": state.time, "x": state.location.x, "y": state.location.y} for state in path]
        return output

    @staticmethod
    def node_signature(node):
        return tuple(sorted((constraint.higher, constraint.lower) for constraint in node.priority_constraints))

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

    # Searching
    cbswp = CBSWP(env)
    solution = cbswp.search()
    if not solution:
        print("Solution not found")
        return

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_yaml_cost(solution)
    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)

if __name__ == "__main__":
    main()