import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import re

def main():
    grid = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    path = []
    # Read the output text
    try:
        with open('mstar_output.txt', 'r') as f:
            for line in f:
                if line.startswith('Time'):
                    # Regex to find (X, Y) tuples
                    tup_strs = re.findall(r'\(\d+, \d+\)', line)
                    step_pos = []
                    for ts in tup_strs:
                        x, y = map(int, re.sub(r'[\(\)]', '', ts).split(','))
                        step_pos.append((x, y))
                    path.append(step_pos)
    except FileNotFoundError:
        print("mstar_output.txt not found. Did you run the basic_mstar.py script and save the output?")
        return
        
    if not path:
        print("No path data found in mstar_output.txt")
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Setting limits. We'll plot X horizontal, Y vertical (downwards like grid matrix)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(4.5, -0.5) 
    
    # Hide axis ticks but show grid
    ax.set_xticks(np.arange(-0.5, 5.5, 1))
    ax.set_yticks(np.arange(-0.5, 5.5, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='black', linestyle='-', linewidth=2)

    # Draw obstacles
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            if cell == 1:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='red'))

    # Initialize agents (Agent 1 is orange/blue, Agent 2 is green)
    colors = ['orange', 'green']
    agents = []
    agent_labels = []
    num_agents = len(path[0])
    
    for i in range(num_agents):
        x, y = path[0][i]
        circle = plt.Circle((x, y), 0.35, color=colors[i], zorder=4)
        ax.add_patch(circle)
        agents.append(circle)
        
        # Add a text label inside
        label = ax.text(x, y, str(i), color='white', ha='center', va='center', fontweight='bold', zorder=5)
        agent_labels.append(label)

    # Add goals with borders
    goals = [(4, 4), (0, 4)]
    for i in range(num_agents):
        gx, gy = goals[i]
        ax.add_patch(plt.Rectangle((gx-0.4, gy-0.4), 0.8, 0.8, fill=False, edgecolor=colors[i], linewidth=3, linestyle='--', zorder=3))

    def update(frame):
        for i, (a, label) in enumerate(zip(agents, agent_labels)):
            x, y = path[frame][i]
            a.center = (x, y)
            label.set_position((x, y))
        return agents + agent_labels

    print("Generating video... This may take a few seconds.")
    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=800, blit=True)
    
    output_filename = 'mstar_video.gif'
    ani.save(output_filename, writer='pillow', fps=2)
    print(f"Animation successfully saved to {output_filename}!")

if __name__ == "__main__":
    main()
