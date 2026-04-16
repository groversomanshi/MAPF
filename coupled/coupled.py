import argparse
import yaml
import heapq
from math import fabs


class Location(object):
    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return str((self.x, self.y))


class JointState:
    """Represents the configuration of ALL agents at one timestep"""
    def __init__(self, locations):
        # 'locations' is a tuple or list of Location objects, one for each agent.
        # We store it as a tuple because tuples are immutable, which is needed for hashing.
        self.locations = tuple(locations)

    def __eq__(self, other):
        # Two joint states are equal ONLY if every agent is in the exact same spot
        if not isinstance(other, JointState): 
            return False
        return self.locations == other.locations

    def __hash__(self):
        # We need a hash function so JointState can be used in sets and dicts
        return hash(tuple((loc.x, loc.y) for loc in self.locations))
    
    def __lt__(self, other):
        # For priority queue comparisons
        return hash(self) < hash(other)
    
    def __str__(self):
        return "[" + ", ".join([str(loc) for loc in self.locations]) + "]"


class Environment(object):
    """
    Environment for coupled algorithm.
    
    Handles:
    - Grid map and obstacles
    - Agent start/goal positions
    - Neighbor generation for single agent locations
    - Collision detection
    """
    
    def __init__(self, dimension, agents, obstacles):
        """
        Args:
            dimension: [width, height] of grid
            agents: List of agent dicts with 'name', 'start', 'goal'
                   Example: [{'name': 'agent_0', 'start': [0, 0], 'goal': [4, 4]}, ...]
            obstacles: List of obstacle coordinates [(x,y), ...]
        """
        self.dimension = dimension
        self.obstacles = set(obstacles)  # Set for fast lookup
        self.agents = agents
        self.agent_dict = {}
        self.make_agent_dict()

    def make_agent_dict(self):
        """Create dictionary mapping agent name to start/goal locations"""
        for agent in self.agents:
            name = agent['name']
            start_loc = Location(agent['start'][0], agent['start'][1])
            goal_loc = Location(agent['goal'][0], agent['goal'][1])
            
            self.agent_dict[name] = {
                'start': start_loc,
                'goal': goal_loc
            }

    def get_neighbors(self, location):
        """
        Get valid neighbors for a single agent from a location.
        
        Returns list of Location objects the agent can move to (including staying in place).
        """
        neighbors = []
        x, y = location.x, location.y
        
        # Wait (stay in same location)
        if self.is_valid_location(Location(x, y)):
            neighbors.append(Location(x, y))
        
        # Move up
        if self.is_valid_location(Location(x, y + 1)):
            neighbors.append(Location(x, y + 1))
        
        # Move down
        if self.is_valid_location(Location(x, y - 1)):
            neighbors.append(Location(x, y - 1))
        
        # Move left
        if self.is_valid_location(Location(x - 1, y)):
            neighbors.append(Location(x - 1, y))
        
        # Move right
        if self.is_valid_location(Location(x + 1, y)):
            neighbors.append(Location(x + 1, y))
        
        return neighbors

    def is_valid_location(self, location):
        """Check if a location is within bounds and obstacle-free"""
        x, y = location.x, location.y
        
        # Check bounds
        if x < 0 or x >= self.dimension[0] or y < 0 or y >= self.dimension[1]:
            return False
        
        # Check obstacles
        if (x, y) in self.obstacles:
            return False
        
        return True

    def get_num_agents(self):
        """Return number of agents"""
        return len(self.agents)
    
    def get_agent_names(self):
        """Return list of agent names"""
        return [agent['name'] for agent in self.agents]
    
    def get_agent_start(self, agent_name):
        """Get start location for an agent"""
        return self.agent_dict[agent_name]['start']
    
    def get_agent_goal(self, agent_name):
        """Get goal location for an agent"""
        return self.agent_dict[agent_name]['goal']


class CollisionChecker:
    """Handles collision detection in joint state space"""
    
    @staticmethod
    def has_vertex_collision(joint_state):
        """
        Check if two agents occupy the same cell at same time (vertex collision).
        
        Args:
            joint_state: JointState object
            
        Returns:
            True if collision detected, False otherwise
        """
        locations = joint_state.locations
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                if locations[i] == locations[j]:
                    return True
        return False
    
    @staticmethod
    def has_edge_collision(prev_state, curr_state):
        """
        Check if two agents swap positions (edge collision).
        
        Example: Agent A goes from (0,0) → (1,0) while Agent B goes from (1,0) → (0,0)
        
        Args:
            prev_state: Previous JointState
            curr_state: Current JointState
            
        Returns:
            True if collision detected, False otherwise
        """
        prev_locs = prev_state.locations
        curr_locs = curr_state.locations
        
        for i in range(len(prev_locs)):
            for j in range(i + 1, len(prev_locs)):
                # If agents swapped positions
                if (prev_locs[i] == curr_locs[j] and 
                    prev_locs[j] == curr_locs[i]):
                    return True
        return False
    
    @staticmethod
    def is_valid_move(prev_state, curr_state):
        """
        Check if transition from prev_state to curr_state is collision-free.
        
        Returns False if ANY collision detected.
        """
        # Check vertex collision (both agents at same spot)
        if CollisionChecker.has_vertex_collision(curr_state):
            return False
        
        # Check edge collision (agents swapping)
        if CollisionChecker.has_edge_collision(prev_state, curr_state):
            return False
        
        return True





class CoupledPlanner:
    """
    Coupled algorithm for multi-agent path planning.
    
    Uses A* search in the joint state space to find collision-free paths
    for all agents simultaneously.
    """
    
    def __init__(self, environment):
        """
        Args:
            environment: Environment object with map, agents, obstacles
        """
        self.env = environment
        self.collision_checker = CollisionChecker()
    
    def get_joint_neighbors(self, joint_state):
        """
        ⭐ CRITICAL FUNCTION ⭐
        Generate all VALID neighboring joint states from current joint state.
        
        This is where multi-agent coordination happens!
        
        Algorithm:
        1. For each agent, get all possible moves (including staying in place)
        2. Generate Cartesian product of all individual moves
        3. Filter out moves with collisions
        4. Return valid joint moves
        
        Returns:
            List of valid JointState objects
        """
        agent_names = self.env.get_agent_names()
        num_agents = len(agent_names)
        
        # Get possible moves for each agent
        individual_moves = []
        for i in range(num_agents):
            agent_loc = joint_state.locations[i]
            moves = self.env.get_neighbors(agent_loc)
            individual_moves.append(moves)
        
        # Generate Cartesian product of all moves
        joint_neighbors = []
        self._generate_joint_moves(individual_moves, 0, [], joint_neighbors, joint_state)
        
        return joint_neighbors
    
    def _generate_joint_moves(self, individual_moves, agent_idx, current_move, 
                             result, prev_joint_state):
        """
        Recursively generate Cartesian product of individual moves.
        Filters out invalid moves (collisions) EARLY to prune the search.
        """
        if agent_idx == len(individual_moves):
            # We've made a valid move for all agents
            result.append(JointState(current_move[:]))
            return
        
        # Try each possible move for agent_idx
        for move in individual_moves[agent_idx]:
            # --- EARLY COLLISION CHECK ---
            collision = False
            
            # 1. Vertex collision: check if this move collides with agents already placed in current_move
            for i in range(agent_idx):
                if move == current_move[i]:
                    collision = True
                    break
            
            if collision: continue

            # 2. Edge collision: check if agent_idx swaps with any agent already placed
            for i in range(agent_idx):
                # prev_agent_i -> curr_agent_i (current_move[i])
                # prev_agent_idx -> curr_agent_idx (move)
                # Swap if: prev_agent_i == move AND prev_agent_idx == current_move[i]
                if (prev_joint_state.locations[i] == move and 
                    prev_joint_state.locations[agent_idx] == current_move[i]):
                    collision = True
                    break
            
            if collision: continue
            
            # Move is valid so far, proceed to next agent
            current_move.append(move)
            self._generate_joint_moves(individual_moves, agent_idx + 1, current_move, result, prev_joint_state)
            current_move.pop()
    
    def heuristic(self, joint_state):
        """
        Admissible heuristic for joint state.
        
        IMPORTANT: Must be admissible (never overestimate true cost)!
        
        Using: Sum of Manhattan distances to goals
        This is admissible because each agent's true cost >= its distance to goal.
        """
        agent_names = self.env.get_agent_names()
        total_distance = 0
        
        for i, agent_name in enumerate(agent_names):
            current_loc = joint_state.locations[i]
            goal_loc = self.env.get_agent_goal(agent_name)
            
            # Manhattan distance
            distance = abs(current_loc.x - goal_loc.x) + abs(current_loc.y - goal_loc.y)
            total_distance += distance
        
        return total_distance
    
    def is_goal_state(self, joint_state):
        """Check if all agents have reached their goals"""
        agent_names = self.env.get_agent_names()
        
        for i, agent_name in enumerate(agent_names):
            if joint_state.locations[i] != self.env.get_agent_goal(agent_name):
                return False
        return True
    
    def reconstruct_path(self, came_from, current):
        """Reconstruct path from dict of parent pointers"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
    
    def solve(self):
        """
        Main A* search in joint state space.
        
        Returns:
            List of JointState objects from start to goal, or None if no solution
        """
        agent_names = self.env.get_agent_names()
        
        # Create initial joint state (all agents at start positions)
        start_locations = [self.env.get_agent_start(name) for name in agent_names]
        initial_state = JointState(start_locations)
        
        # Create goal state (all agents at goal positions)
        goal_locations = [self.env.get_agent_goal(name) for name in agent_names]
        goal_state = JointState(goal_locations)
        
        # A* data structures
        open_set = []
        heapq.heappush(open_set, (0, initial_state))
        
        came_from = {}
        g_score = {initial_state: 0}
        f_score = {initial_state: self.heuristic(initial_state)}
        
        closed_set = set()
        
        step_cost = 1
        max_iterations = 100000  # Safety limit
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # Get state with lowest f_score
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            # Check if we reached the goal
            if self.is_goal_state(current):
                print(f"✓ Goal reached in {iterations} iterations!")
                return self.reconstruct_path(came_from, current)
            
            closed_set.add(current)
            
            # Explore neighbors
            neighbors = self.get_joint_neighbors(current)
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                tentative_g = g_score.get(current, float('inf')) + step_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print(f"✗ No solution found after {iterations} iterations")
        return None


class SolutionValidator:
    """Validates that a solution path is collision-free"""
    
    @staticmethod
    def validate(solution_path, env):
        """
        Check solution for:
        1. Collision-free at each timestep (vertex collision)
        2. No agents swapping (edge collision)
        3. All agents at goals at end
        
        Returns: (is_valid, error_message)
        """
        collision_checker = CollisionChecker()
        
        # Check each state for vertex collisions
        for t, joint_state in enumerate(solution_path):
            if collision_checker.has_vertex_collision(joint_state):
                return False, f"Vertex collision at step {t}: {joint_state}"
        
        # Check transitions for edge collisions
        for t in range(len(solution_path) - 1):
            if not collision_checker.is_valid_move(solution_path[t], solution_path[t + 1]):
                return False, f"Edge collision between step {t} and {t+1}"
        
        # Check that all agents at goals
        agent_names = env.get_agent_names()
        final_state = solution_path[-1]
        
        for i, agent_name in enumerate(agent_names):
            if final_state.locations[i] != env.get_agent_goal(agent_name):
                return False, f"Agent {agent_name} not at goal: {final_state.locations[i]} vs {env.get_agent_goal(agent_name)}"
        
        return True, "Solution is valid!"


def solution_to_yaml(solution_path, env, output_file):
    """
    Convert solution (list of JointStates) to YAML format for visualization.
    
    Output format matches what visualize.py expects.
    """
    agent_names = env.get_agent_names()
    schedule = {"schedule": {}}
    
    for agent_name in agent_names:
        schedule["schedule"][agent_name] = []
    
    for t, joint_state in enumerate(solution_path):
        for i, agent_name in enumerate(agent_names):
            loc = joint_state.locations[i]
            schedule["schedule"][agent_name].append({
                "x": loc.x,
                "y": loc.y,
                "t": t
            })
    
    with open(output_file, 'w') as f:
        yaml.dump(schedule, f)
    
    print(f"✓ Schedule saved to {output_file}")


def create_input_yaml(env, output_file):
    """Create the input map YAML file for visualization"""
    agent_names = env.get_agent_names()
    
    agents_list = []
    for agent_name in agent_names:
        start = env.get_agent_start(agent_name)
        goal = env.get_agent_goal(agent_name)
        agents_list.append({
            "name": agent_name,
            "start": [start.x, start.y],
            "goal": [goal.x, goal.y]
        })
    
    map_data = {
        "agents": agents_list,
        "map": {
            "dimensions": env.dimension,
            "obstacles": [[x, y] for x, y in env.obstacles]
        }
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(map_data, f)
    
    print(f"✓ Map saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coupled Algorithm for Multi-Agent Path Planning")
    parser.add_argument("input", nargs="?", help="Input YAML file")
    parser.add_argument("output", nargs="?", help="Output schedule YAML file")
    args = parser.parse_args()

    if args.input:
        print(f"Loading test case from {args.input}...")
        with open(args.input, 'r') as f:
            data = yaml.safe_load(f)
        
        dimension = data['map']['dimensions']
        agents = data['agents']
        obstacles = [tuple(o) for o in data['map']['obstacles']]
        
        print("\n→ Creating environment...")
        env = Environment(dimension, agents, obstacles)
        print(f"  Grid: {dimension}, Agents: {len(agents)}, Obstacles: {len(obstacles)}")
        
        print("\n→ Running coupled planner...")
        planner = CoupledPlanner(env)
        solution = planner.solve()
        
        if solution:
            print(f"\n→ Solution length: {len(solution)} timesteps")
            
            print("\n→ Validating solution...")
            is_valid, msg = SolutionValidator.validate(solution, env)
            print(f"  {msg}")
            
            if is_valid:
                if args.output:
                    solution_to_yaml(solution, env, args.output)
                    print(f"\n✓ Success! Solution saved to {args.output}")
                else:
                    print("\n✓ Success! (No output file specified)")
        else:
            print("\n✗ No solution found")
            
    else:
        print("Coupled Algorithm for Multi-Agent Path Planning")
        print("=" * 60)
        
        # Test case 1: 2 agents on 5x5 grid WITH obstacles
        dimension = [5, 5]
        agents = [
            {'name': 'agent_0', 'start': [0, 0], 'goal': [4, 4]},
            {'name': 'agent_1', 'start': [4, 0], 'goal': [0, 4]},
        ]
        obstacles = [(2, 1), (2, 2), (2, 3)]  # Vertical wall obstacle
        
        print("\n→ Test 1: Crossing with Obstacles")
        print("=" * 60)
        
        print("\n→ Creating environment...")
        env = Environment(dimension, agents, obstacles)
        print(f"  Grid: {dimension}, Agents: {len(agents)}, Obstacles: {len(obstacles)}")
        print(f"  Obstacles at: {obstacles}")
        
        print("\n→ Running coupled planner...")
        planner = CoupledPlanner(env)
        solution = planner.solve()
        
        if solution:
            print(f"\n→ Solution length: {len(solution)} timesteps")
            
            print("\n→ Validating solution...")
            is_valid, msg = SolutionValidator.validate(solution, env)
            print(f"  {msg}")
            
            if is_valid:
                print("\n→ Exporting to YAML...")
                create_input_yaml(env, "test1_input.yaml")
                solution_to_yaml(solution, env, "test1_schedule.yaml")
                
                print("\n✓ Test 1 Success! To visualize:")
                print("  python3 visualize_coupled.py test1_input.yaml test1_schedule.yaml --video test1_solution.gif")
        else:
            print("\n✗ No solution found")
        
        # Test case 2: 2 agents WITH MORE obstacles forcing them to coordinate
        print("\n\n→ Test 2: Complex Obstacles")
        print("=" * 60)
        
        dimension = [8, 8]
        agents = [
            {'name': 'agent_0', 'start': [0, 0], 'goal': [7, 7]},
            {'name': 'agent_1', 'start': [7, 0], 'goal': [0, 7]},
        ]
        # Add a wall of obstacles in the middle
        obstacles = [
            (3, 3), (3, 4), (3, 5),
            (4, 3), (4, 5),
            (5, 3), (5, 4), (5, 5),
        ]
        
        print("\n→ Creating environment with obstacles...")
        env = Environment(dimension, agents, obstacles)
        print(f"  Grid: {dimension}, Agents: {len(agents)}, Obstacles: {len(obstacles)}")
        print(f"  Obstacles forming wall pattern")
        
        print("\n→ Running coupled planner...")
        planner = CoupledPlanner(env)
        solution = planner.solve()
        
        if solution:
            print(f"\n→ Solution length: {len(solution)} timesteps")
            
            print("\n→ Validating solution...")
            is_valid, msg = SolutionValidator.validate(solution, env)
            print(f"  {msg}")
            
            if is_valid:
                print("\n→ Exporting to YAML...")
                create_input_yaml(env, "test2_input.yaml")
                solution_to_yaml(solution, env, "test2_schedule.yaml")
                
                print("\n✓ Test 2 Success! To visualize:")
                print("  python3 visualize_coupled.py test2_input.yaml test2_schedule.yaml --video test2_solution.gif")
        else:
            print("\n✗ No solution found for obstacles test")

