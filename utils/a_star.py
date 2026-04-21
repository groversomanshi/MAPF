import heapq
import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    x: int
    y: int


@dataclass(frozen=True)
class State:
    time: int
    location: Location

    def is_equal_except_time(self, other):
        return self.location == other.location


@dataclass(frozen=True)
class VertexConstraint:
    agent: str
    time: int
    location: Location


@dataclass(frozen=True)
class EdgeConstraint:
    agent: str
    time: int
    location_1: Location
    location_2: Location


class SingleAgentAStar:
    def __init__(self, dimension, obstacles):
        self.dimension = dimension
        self.obstacles = set(map(tuple, obstacles))

    def search(self, agent, start, goal, constraints=None, reservation_table=None):
        # plain old a* for one agent under cbs-style constraints
        constraints = constraints or []
        reservation_table = reservation_table or {}

        if not self._valid_state(agent, State(0, start), constraints, reservation_table):
            return None
        
        open_list = []
        counter = itertools.count()
        start_state = State(0, start)
        start_key = (start_state.time, start_state.location)
        g_score = {start_key: 0}
        parents = {start_key: None}

        heapq.heappush(
            open_list,
            (
                self._heuristic(start, goal),
                next(counter),
                start_state,
            ),
        )

        max_constraint_time = self._max_constraint_time(
            agent,
            goal,
            constraints,
            reservation_table,
        )

        while open_list:
            _, _, current = heapq.heappop(open_list)
            current_key = (current.time, current.location)

            # stay alive long enough to clear any future constraints at the goal
            if current.location == goal and current.time >= max_constraint_time:
                return self._reconstruct_path(current_key, parents)

            for neighbor in self._neighbors(current):
                if not self._valid_state(agent, neighbor, constraints, reservation_table):
                    continue
                if not self._valid_transition(agent, current, neighbor, constraints, reservation_table):
                    continue

                neighbor_key = (neighbor.time, neighbor.location)
                tentative_g = g_score[current_key] + 1
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                g_score[neighbor_key] = tentative_g
                parents[neighbor_key] = current_key
                f_score = tentative_g + self._heuristic(neighbor.location, goal)
                heapq.heappush(open_list, (f_score, next(counter), neighbor))

        return None

    def _neighbors(self, state):
        moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]
        neighbors = []
        for dx, dy in moves:
            nx = state.location.x + dx
            ny = state.location.y + dy
            if 0 <= nx < self.dimension[0] and 0 <= ny < self.dimension[1]:
                if (nx, ny) not in self.obstacles:
                    neighbors.append(State(state.time + 1, Location(nx, ny)))
        return neighbors

    @staticmethod
    def _heuristic(location, goal):
        return abs(location.x - goal.x) + abs(location.y - goal.y)

    @staticmethod
    def _max_constraint_time(agent, goal, constraints, reservation_table):
        max_time = 0
        for constraint in constraints:
            if constraint.agent != agent:
                continue
            if isinstance(constraint, VertexConstraint) and constraint.location == goal:
                max_time = max(max_time, constraint.time + 1)
            elif (
                isinstance(constraint, EdgeConstraint)
                and constraint.location_1 == goal
                and constraint.location_2 == goal
            ):
                max_time = max(max_time, constraint.time + 1)
            
        if reservation_table:
            res_times = [t for t, loc in reservation_table.keys() if loc == (goal.x, goal.y)]
            if res_times:
                max_time = max(max_time, max(res_times) + 1)
        return max_time

    @staticmethod
    def _valid_state(agent, state, constraints, reservation_table):
        vertex_constraint = VertexConstraint(agent, state.time, state.location)
        if vertex_constraint in constraints:
            return False
            
        loc = (state.location.x, state.location.y)
        if (state.time, loc) in reservation_table:
            return False
            
        return True

    @staticmethod
    def _valid_transition(agent, previous, current, constraints, reservation_table):
        edge_constraint = EdgeConstraint(
            agent,
            previous.time,
            previous.location,
            current.location,
        )
        if edge_constraint in constraints:
            return False
            
        loc1 = (previous.location.x, previous.location.y)
        loc2 = (current.location.x, current.location.y)
        t2 = current.time
        
        if (t2, loc1) in reservation_table and (t2 - 1, loc2) in reservation_table:
            for occ in reservation_table.get((t2, loc1), []):
                if occ in reservation_table.get((t2 - 1, loc2), []):
                    return False
        
        return True

    @staticmethod
    def _reconstruct_path(goal_key, parents):
        path = []
        current_key = goal_key
        while current_key is not None:
            time, location = current_key
            path.append(State(time, location))
            current_key = parents[current_key]
        path.reverse()
        return path


class JointAgentAStar:
    def __init__(self, dimension, obstacles):
        self.dimension = dimension
        self.obstacles = set(map(tuple, obstacles))
        self.moves = [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0)]

    def search(self, agents, starts, goals, constraints=None):
        constraints = constraints or []
        
        # this is the coupled planner used after a merge
        agents = tuple(agents)
        start_locations = tuple(starts[agent] for agent in agents)
        goal_locations = tuple(goals[agent] for agent in agents)

        if not self._valid_joint_state(agents, 0, start_locations, constraints):
            return None
        if self._has_internal_conflict(start_locations, start_locations):
            return None

        open_list = []
        counter = itertools.count()
        start_key = (0, start_locations)
        g_score = {start_key: 0}
        parents = {start_key: None}

        heapq.heappush(
            open_list,
            (
                self._heuristic(start_locations, goal_locations),
                next(counter),
                start_key,
            ),
        )

        max_constraint_time = self._max_constraint_time(agents, goal_locations, constraints)

        while open_list:
            _, _, current_key = heapq.heappop(open_list)
            time, current_locations = current_key

            # same goal rule as single-agent a*: don't stop before future constraints expire
            if current_locations == goal_locations and time >= max_constraint_time:
                return self._reconstruct_paths(agents, current_key, parents)

            for next_locations in self._joint_neighbors(current_locations):
                next_time = time + 1
                if not self._valid_joint_state(agents, next_time, next_locations, constraints):
                    continue
                if not self._valid_joint_transition(
                    agents,
                    time,
                    current_locations,
                    next_locations,
                    constraints,
                ):
                    continue
                # merged agents still cannot collide with each other internally
                if self._has_internal_conflict(current_locations, next_locations):
                    continue

                neighbor_key = (next_time, next_locations)
                tentative_g = g_score[current_key] + len(agents)
                if tentative_g >= g_score.get(neighbor_key, float("inf")):
                    continue

                g_score[neighbor_key] = tentative_g
                parents[neighbor_key] = current_key
                f_score = tentative_g + self._heuristic(next_locations, goal_locations)
                heapq.heappush(open_list, (f_score, next(counter), neighbor_key))

        return None

    def _joint_neighbors(self, current_locations):
        # build the cartesian product of each agent's legal moves
        candidate_moves = []
        for location in current_locations:
            options = []
            for dx, dy in self.moves:
                nx = location.x + dx
                ny = location.y + dy
                if 0 <= nx < self.dimension[0] and 0 <= ny < self.dimension[1]:
                    if (nx, ny) not in self.obstacles:
                        options.append(Location(nx, ny))
            candidate_moves.append(options)
        return itertools.product(*candidate_moves)

    @staticmethod
    def _heuristic(locations, goals):
        return sum(
            abs(location.x - goal.x) + abs(location.y - goal.y)
            for location, goal in zip(locations, goals)
        )

    @staticmethod
    def _max_constraint_time(agents, goals, constraints):
        goal_by_agent = dict(zip(agents, goals))
        max_time = 0
        for constraint in constraints:
            goal = goal_by_agent.get(constraint.agent)
            if goal is None:
                continue
            if isinstance(constraint, VertexConstraint) and constraint.location == goal:
                max_time = max(max_time, constraint.time + 1)
            elif (
                isinstance(constraint, EdgeConstraint)
                and constraint.location_1 == goal
                and constraint.location_2 == goal
            ):
                max_time = max(max_time, constraint.time + 1)
        return max_time

    @staticmethod
    def _valid_joint_state(agents, time, locations, constraints):
        for agent, location in zip(agents, locations):
            if VertexConstraint(agent, time, location) in constraints:
                return False
        return True

    @staticmethod
    def _valid_joint_transition(agents, time, previous_locations, next_locations, constraints):
        for agent, previous, current in zip(agents, previous_locations, next_locations):
            if EdgeConstraint(agent, time, previous, current) in constraints:
                return False
        return True

    @staticmethod
    def _has_internal_conflict(previous_locations, next_locations):
        if len(set(next_locations)) != len(next_locations):
            return True

        for i in range(len(previous_locations)):
            for j in range(i + 1, len(previous_locations)):
                if (
                    previous_locations[i] == next_locations[j]
                    and previous_locations[j] == next_locations[i]
                ):
                    return True
        return False

    @staticmethod
    def _reconstruct_paths(agents, goal_key, parents):
        sequence = []
        current_key = goal_key
        while current_key is not None:
            sequence.append(current_key)
            current_key = parents[current_key]
        sequence.reverse()

        paths = {agent: [] for agent in agents}
        for time, locations in sequence:
            for agent, location in zip(agents, locations):
                paths[agent].append(State(time, location))
        return paths
