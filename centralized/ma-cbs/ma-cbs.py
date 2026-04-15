import argparse
import heapq
import itertools
import yaml

try:
    from .a_star import (
        EdgeConstraint,
        JointAgentAStar,
        Location,
        SingleAgentAStar,
        State,
        VertexConstraint,
    )
except ImportError:
    from a_star import (
        EdgeConstraint,
        JointAgentAStar,
        Location,
        SingleAgentAStar,
        State,
        VertexConstraint,
    )


class Conflict:
    VERTEX = "vertex"
    EDGE = "edge"

    def __init__(
        self,
        time,
        conflict_type,
        agent_1,
        agent_2,
        location_1,
        location_2=None,
        edge_1=None,
        edge_2=None,
    ):
        self.time = time
        self.type = conflict_type
        self.agent_1 = agent_1
        self.agent_2 = agent_2
        self.location_1 = location_1
        self.location_2 = location_2
        self.edge_1 = edge_1
        self.edge_2 = edge_2


class HighLevelNode:
    def __init__(self):
        self.groups = []
        self.solution = {}
        self.constraints = set()
        self.cost = 0
        # tracks how many times two groups have fought so we know when to merge them
        self.merge_counts = {}

    def copy(self):
        node = HighLevelNode()
        node.groups = [frozenset(group) for group in self.groups]
        node.solution = {agent: list(path) for agent, path in self.solution.items()}
        node.constraints = set(self.constraints)
        node.cost = self.cost
        node.merge_counts = dict(self.merge_counts)
        return node


class MACBSEnvironment:
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        self.obstacles = set(map(tuple, obstacles))
        self.agents = agents
        self.agent_dict = {}
        self._make_agent_dict()
        # use single-agent a* before a merge and joint a* after a merge
        self.single_planner = SingleAgentAStar(dimension, obstacles)
        self.joint_planner = JointAgentAStar(dimension, obstacles)

    def _make_agent_dict(self):
        for agent in self.agents:
            self.agent_dict[agent["name"]] = {
                "start": Location(*agent["start"]),
                "goal": Location(*agent["goal"]),
            }

    def initial_groups(self):
        return [frozenset([agent["name"]]) for agent in self.agents]

    def plan_for_group(self, group, constraints):
        if len(group) == 1:
            agent = next(iter(group))
            path = self.single_planner.search(
                agent,
                self.agent_dict[agent]["start"],
                self.agent_dict[agent]["goal"],
                constraints,
            )
            if path is None:
                return None
            return {agent: path}

        # once a group has more than one agent, solve it as one coupled problem
        ordered_agents = tuple(sorted(group))
        starts = {agent: self.agent_dict[agent]["start"] for agent in ordered_agents}
        goals = {agent: self.agent_dict[agent]["goal"] for agent in ordered_agents}
        return self.joint_planner.search(ordered_agents, starts, goals, constraints)

    @staticmethod
    def compute_solution_cost(solution):
        return sum(len(path) for path in solution.values())

    @staticmethod
    def get_state(agent_name, solution, time):
        path = solution[agent_name]
        if time < len(path):
            return path[time]
        return path[-1]

    def get_first_conflict(self, solution, groups):
        group_lookup = {}
        for group in groups:
            for agent in group:
                group_lookup[agent] = group

        max_t = max(len(path) for path in solution.values())
        agents = sorted(solution.keys())
        for time in range(max_t):
            # skip conflicts inside the same merged group because the joint low-level
            # planner already guarantees those are clean
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    agent_1 = agents[i]
                    agent_2 = agents[j]
                    if group_lookup[agent_1] == group_lookup[agent_2]:
                        continue

                    state_1 = self.get_state(agent_1, solution, time)
                    state_2 = self.get_state(agent_2, solution, time)
                    if state_1.is_equal_except_time(state_2):
                        return Conflict(
                            time,
                            Conflict.VERTEX,
                            agent_1,
                            agent_2,
                            state_1.location,
                        )

            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    agent_1 = agents[i]
                    agent_2 = agents[j]
                    if group_lookup[agent_1] == group_lookup[agent_2]:
                        continue

                    state_1a = self.get_state(agent_1, solution, time)
                    state_1b = self.get_state(agent_1, solution, time + 1)
                    state_2a = self.get_state(agent_2, solution, time)
                    state_2b = self.get_state(agent_2, solution, time + 1)

                    if (
                        state_1a.is_equal_except_time(state_2b)
                        and state_1b.is_equal_except_time(state_2a)
                    ):
                        return Conflict(
                            time,
                            Conflict.EDGE,
                            agent_1,
                            agent_2,
                            state_1a.location,
                            state_1b.location,
                            edge_1=(state_1a.location, state_1b.location),
                            edge_2=(state_2a.location, state_2b.location),
                        )
        return None

    @staticmethod
    def create_constraint_from_conflict(conflict, agent):
        if conflict.type == Conflict.VERTEX:
            return VertexConstraint(agent, conflict.time, conflict.location_1)
        if agent == conflict.agent_1:
            location_1, location_2 = conflict.edge_1
        elif agent == conflict.agent_2:
            location_1, location_2 = conflict.edge_2
        else:
            raise KeyError(f"Agent {agent} is not part of the conflict")
        return EdgeConstraint(
            agent,
            conflict.time,
            location_1,
            location_2,
        )

    @staticmethod
    def group_signature(groups):
        return tuple(sorted(tuple(sorted(group)) for group in groups))

    @staticmethod
    def constraints_signature(constraints):
        signature = []
        for constraint in constraints:
            if isinstance(constraint, VertexConstraint):
                signature.append(
                    (
                        "v",
                        constraint.agent,
                        constraint.time,
                        constraint.location.x,
                        constraint.location.y,
                    )
                )
            else:
                signature.append(
                    (
                        "e",
                        constraint.agent,
                        constraint.time,
                        constraint.location_1.x,
                        constraint.location_1.y,
                        constraint.location_2.x,
                        constraint.location_2.y,
                    )
                )
        return tuple(sorted(signature))


class MACBS:
    def __init__(self, environment, merge_bound):
        self.env = environment
        self.merge_bound = merge_bound
        self.open_list = []
        self.counter = itertools.count()

    def search(self):
        root = HighLevelNode()
        root.groups = self.env.initial_groups()
        root.solution = self._compute_solution_for_groups(root.groups, root.constraints)
        if root.solution is None:
            return {}
        root.cost = self.env.compute_solution_cost(root.solution)
        self._push(root)

        closed = set()

        while self.open_list:
            _, _, node = heapq.heappop(self.open_list)
            node_signature = self._node_signature(node)
            if node_signature in closed:
                continue
            closed.add(node_signature)

            conflict = self.env.get_first_conflict(node.solution, node.groups)
            if conflict is None:
                print("solution found")
                return self._generate_plan(node.solution)

            group_1 = self._find_group(node.groups, conflict.agent_1)
            group_2 = self._find_group(node.groups, conflict.agent_2)
            pair_key = tuple(sorted((tuple(sorted(group_1)), tuple(sorted(group_2)))))
            conflict_count = node.merge_counts.get(pair_key, 0) + 1

            # if these groups keep colliding, stop branching and just merge them
            if conflict_count > self.merge_bound:
                merged_node = self._merge_groups(node, group_1, group_2, pair_key, conflict_count)
                if merged_node is not None:
                    self._push(merged_node)
                continue

            # otherwise do the normal cbs split: constrain one side, then the other
            for agent in (conflict.agent_1, conflict.agent_2):
                child = self._branch_on_conflict(node, conflict, agent, pair_key, conflict_count)
                if child is not None:
                    self._push(child)

        return {}

    def _compute_solution_for_groups(self, groups, constraints, existing_solution=None, target_group=None):
        solution = {}
        if existing_solution is not None:
            # keep the unaffected paths and only replan what changed
            solution = {agent: list(path) for agent, path in existing_solution.items()}

        groups_to_plan = groups if target_group is None else [target_group]
        for group in groups_to_plan:
            group_solution = self.env.plan_for_group(group, constraints)
            if group_solution is None:
                return None
            solution.update(group_solution)

        return solution

    def _merge_groups(self, node, group_1, group_2, pair_key, conflict_count):
        child = node.copy()
        merged_group = frozenset(group_1 | group_2)
        child.groups = [group for group in child.groups if group not in (group_1, group_2)]
        child.groups.append(merged_group)
        child.merge_counts[pair_key] = conflict_count
        # after a merge, only the new meta-agent group needs a fresh low-level solve
        child.solution = self._compute_solution_for_groups(
            child.groups,
            child.constraints,
            existing_solution=child.solution,
            target_group=merged_group,
        )
        if child.solution is None:
            return None
        child.cost = self.env.compute_solution_cost(child.solution)
        return child

    def _branch_on_conflict(self, node, conflict, agent, pair_key, conflict_count):
        child = node.copy()
        child.merge_counts[pair_key] = conflict_count
        child.constraints.add(self.env.create_constraint_from_conflict(conflict, agent))

        target_group = self._find_group(child.groups, agent)
        # if the agent already belongs to a merged group, replan that whole group together
        child.solution = self._compute_solution_for_groups(
            child.groups,
            child.constraints,
            existing_solution=child.solution,
            target_group=target_group,
        )
        if child.solution is None:
            return None
        child.cost = self.env.compute_solution_cost(child.solution)
        return child

    def _push(self, node):
        heapq.heappush(self.open_list, (node.cost, next(self.counter), node))

    def _node_signature(self, node):
        return (
            self.env.group_signature(node.groups),
            self.env.constraints_signature(node.constraints),
            self._merge_counts_signature(node.groups, node.merge_counts),
        )

    @staticmethod
    def _merge_counts_signature(groups, merge_counts):
        active_groups = {tuple(sorted(group)) for group in groups}
        return tuple(
            sorted(
                (group_1, group_2, count)
                for (group_1, group_2), count in merge_counts.items()
                if group_1 in active_groups and group_2 in active_groups
            )
        )

    @staticmethod
    def _find_group(groups, agent):
        for group in groups:
            if agent in group:
                return group
        raise KeyError(f"Agent {agent} not found in any group")

    @staticmethod
    def _generate_plan(solution):
        plan = {}
        for agent, path in solution.items():
            plan[agent] = [
                {"t": state.time, "x": state.location.x, "y": state.location.y}
                for state in path
            ]
        return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    parser.add_argument(
        "--merge-bound",
        type=int,
        default=None,
        help="merge groups after they conflict more than this many times",
    )
    args = parser.parse_args()

    with open(args.param, "r") as param_file:
        param = yaml.load(param_file, Loader=yaml.FullLoader)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]
    merge_bound = args.merge_bound
    if merge_bound is None:
        merge_bound = param.get("merge_bound", 1)

    env = MACBSEnvironment(dimension, agents, obstacles)
    solver = MACBS(env, merge_bound)
    solution = solver.search()

    if not solution:
        print("Solution not found")
        return

    output = {
        "schedule": solution,
        "cost": sum(len(path) for path in solution.values()),
    }

    with open(args.output, "w") as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()
