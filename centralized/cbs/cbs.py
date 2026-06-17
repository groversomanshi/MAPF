"""

Python implementation of Conflict-based search

author: Ashwin Bose (@atb033)

"""
import sys
sys.path.insert(0, '../')
import argparse
import yaml
from math import fabs
from itertools import combinations
import os
import time
import heapq
import itertools

from utils.a_star import SingleAgentAStar, Location as UtilsLoc, VertexConstraint as UtilsVC, EdgeConstraint as UtilsEC

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
        return hash(str(self.time)+str(self.location.x) + str(self.location.y))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y))

class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'

class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'

class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'

class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])

    def copy(self):
        c = Constraints()
        c.vertex_constraints = set(self.vertex_constraints)
        c.edge_constraints = set(self.edge_constraints)
        return c

class Environment(object):
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles

        self.agents = agents
        self.agent_dict = {}

        self.make_agent_dict()

        self.constraints = Constraints()
        self.constraint_dict = {}

    def get_neighbors(self, state):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        return neighbors


    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])
        for t in range(max_t):
            first_edge_conflict = None
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_2a = self.get_state(agent_2, solution, t)

                if state_1a.is_equal_except_time(state_2a):
                    result = Conflict()
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1a.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result
                    
                state_1b = self.get_state(agent_1, solution, t+1)
                state_2b = self.get_state(agent_2, solution, t+1)

                if (state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a)
                    and first_edge_conflict is None):
                    result = Conflict()
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    first_edge_conflict = result

            if first_edge_conflict:
                return first_edge_conflict
        return False

    def count_conflicts(self, solution):
        count = 0
        max_t = max([len(plan) for plan in solution.values()])
        for t in range(max_t):
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_2a = self.get_state(agent_2, solution, t)
                if state_1a.is_equal_except_time(state_2a):
                    count += 1

                state_1b = self.get_state(agent_1, solution, t+1)
                state_2b = self.get_state(agent_2, solution, t+1)
                if (state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a)):
                    count += 1
        return count

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y) not in self.obstacles

    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(state.location.y - goal.location.y)


    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            goal_state = State(0, Location(agent['goal'][0], agent['goal'][1]))

            self.agent_dict.update({agent['name']:{'start':start_state, 'goal':goal_state}})

    def compute_solution(self, parent_path=None, agent_name=None):
        solution = {name: list(path) for name, path in parent_path.items()} if parent_path is not None else {}

        agents_replan = self.agent_dict.keys() if agent_name is None else [agent_name]
        for agent in agents_replan:
            self.constraints = self.constraint_dict.setdefault(agent, Constraints())
            
            converted_constraints = []
            for vc in self.constraints.vertex_constraints:
                converted_constraints.append(UtilsVC(agent, vc.time, UtilsLoc(vc.location.x, vc.location.y)))
            for ec in self.constraints.edge_constraints:
                converted_constraints.append(UtilsEC(agent, ec.time, UtilsLoc(ec.location_1.x, ec.location_1.y), UtilsLoc(ec.location_2.x, ec.location_2.y)))

            a_star = SingleAgentAStar(self.dimension, self.obstacles)
            start = UtilsLoc(self.agent_dict[agent]["start"].location.x, self.agent_dict[agent]["start"].location.y)
            goal = UtilsLoc(self.agent_dict[agent]["goal"].location.x, self.agent_dict[agent]["goal"].location.y)
            
            path = a_star.search(agent, start, goal, converted_constraints)
            if not path:
                return False
                
            converted_path = []
            for s in path:
                converted_path.append(State(s.time, Location(s.location.x, s.location.y)))
                
            solution.update({agent: converted_path})
        return solution

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])

    @staticmethod
    def constraints_signature(constraint_dict):
        signature = []
        for agent in sorted(constraint_dict.keys()):
            c = constraint_dict[agent]
            for vc in c.vertex_constraints:
                signature.append(("v", agent, vc.time, vc.location.x, vc.location.y))
            for ec in c.edge_constraints:
                signature.append(("e", agent, ec.time, ec.location_1.x, ec.location_1.y,
                                ec.location_2.x, ec.location_2.y))
        return tuple(sorted(signature))

class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __lt__(self, other):
        return self.cost < other.cost

    def copy(self):
        node = HighLevelNode()
        node.constraint_dict = {agent: c.copy() for agent, c in self.constraint_dict.items()}
        node.cost = self.cost
        return node

class CBS(object):
    def __init__(self, environment):
        self.env = environment
        self.open_list = []
        self.counter = itertools.count()
        self.use_bypass = False

    def search(self):
        start = HighLevelNode()
        start.constraint_dict = {agent: Constraints() for agent in self.env.agent_dict.keys()}
        start.solution = self.env.compute_solution()
        if not start.solution:
            # DEBUG:
            # print(f"[CBS] FAILED: no single-agent solution found")
            return {}
        start.cost = self.env.compute_solution_cost(start.solution)

        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))

        # DEBUG:
        nodes_expanded = 0
        replans = 0
        search_start = time.perf_counter()

        closed = set()

        while self.open_list:
            _, _, P = heapq.heappop(self.open_list)
            sig = Environment.constraints_signature(P.constraint_dict)
            if sig in closed:
                continue
            closed.add(sig)

            # DEBUG:
            nodes_expanded += 1
            if nodes_expanded % 50 == 0:
                elapsed_time = time.perf_counter() - search_start
                print(f"[CBS] expanded = {nodes_expanded} open = {len(self.open_list)}" 
                      f" cost = {P.cost} elapsed = {elapsed_time:.2f}s")

            self.env.constraint_dict = P.constraint_dict
            conflict = self.env.get_first_conflict(P.solution)
            if not conflict:
                # DEBUG:
                # print(f"[CBS] SOLUTION FOUND"
                #       f" expanded = {nodes_expanded} replans = {replans} cost = {P.cost}"
                #       f" elapsed = {time.perf_counter() - search_start:.2f}s")
                print("solution found")

                return self.generate_plan(P.solution)

            # DEBUG:
            # print(f"[CBS] conflict #{nodes_expanded} found: {conflict}")
            
            constraint_dict = self.env.create_constraints_from_conflict(conflict)
            
            if self.use_bypass:
                parent_conflicts = self.env.count_conflicts(P.solution)
                bypass_node = None
                children = []

            for agent in constraint_dict.keys():
                new_node = P.copy()
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])

                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution(P.solution, agent)
                # DEBUG:
                # replans += 1

                if not new_node.solution:
                    # DEBUG:
                    # print(f"[CBS] branch agent = {agent} failed to find solution")
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)
                               
                # DEBUG:
                # print(f"[CBS] branch agent = {agent} found solution with cost = {new_node.cost}")

                if self.use_bypass:
                    child_conflicts = self.env.count_conflicts(new_node.solution)
                    if new_node.cost == P.cost and child_conflicts < parent_conflicts:
                        bypass_node = new_node
                        break
                    children.append(new_node)
                else:
                    heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))

            if self.use_bypass:
                if bypass_node:
                    heapq.heappush(self.open_list, (bypass_node.cost, next(self.counter), bypass_node))
                else:
                    for child in children:
                        heapq.heappush(self.open_list, (child.cost, next(self.counter), child))

        # DEBUG:
        # print(f"[CBS] NO SOLUTION FOUND"
        #       f" expanded = {nodes_expanded} replans = {replans}"
        #       f" elapsed = {time.perf_counter() - search_start:.2f}s")
        return {}

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [{'t':state.time, 'x':state.location.x, 'y':state.location.y} for state in path]
            plan[agent] = path_dict_list
        return plan


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
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print(" Solution not found" )
        return

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    with open(args.output, 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()