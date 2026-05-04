import random
import stormpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from jamming_overlay import emit_jamming_block, patch_base_for_jamming


def generate_random_prism(M: int = 5, N: int = 6, num_obstacles: int = 3, seed=None,
                          jamming=None):
    """
    Generates a PRISM file with randomly placed obstacles and goal.
    Returns the file path of the generated model.

    If `jamming` is None (default), the emitted file is the plain base
    model — identical to the prior behavior. If `jamming` is a scenario
    dict (see `jamming_overlay.JammingScenario`, e.g. `BASELINE_5x6`), an
    adversarial-jamming module is composed onto the base via
    synchronization on the movement actions, and `crashed` is extended
    to OR in jamming-induced crashes.
    """
    if seed is not None:
        random.seed(seed)
        
    all_coords = [(x, y) for x in range(1, M + 1) for y in range(1, N + 1)]
    all_coords.remove((1, 1)) # Prevent spawning on the drone's start location
    
    # Sample unique coordinates for the goal and obstacles
    sampled = random.sample(all_coords, 1 + num_obstacles)
    goal_pos = sampled[0]
    obstacle_positions = sampled[1:]
    
    # Format the PRISM formulas dynamically
    goal_formula = f"formula goal = x = {goal_pos[0]} & y = {goal_pos[1]};"
    obs_strings = [f"(x = {ox} & y = {oy})" for ox, oy in obstacle_positions]
    crashed_formula = f"formula crashed = {' | '.join(obs_strings)};"

    # The PRISM Template
    prism_template = f"""mdp

// Model parameters 
const double p = 0.1;
const double e = 0.02;
formula a = x * e;

const int M = {M};
const int N = {N};
const double eps = 0.0001;

// --- DYNAMICALLY GENERATED ELEMENTS ---
{goal_formula}
{crashed_formula}
// --------------------------------------

module env
	x : [1..M] init 1;
	y : [1..N] init 1;
	[up]    !crashed & y < N -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (y'=min(y+1,N)) & (x'=max(x-1,1))
			  + [max(eps,p-a), min(p+a,1)] : (y'=min(y+1,N)) & (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : true;
	[right] !crashed & x < M -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : (x'=min(x+1,M)) & (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (x'=max(x-1,1)) 		     
			  + [max(eps,p-a), min(p+a,1)] : true;
	[left] 	!crashed & x > 1 -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (x'=max(x-1,1))
		   	  + [max(eps,p-a), min(p+a,1)] : (x'=max(x-1,1)) & (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (x'=min(x+1,M)) 		     
			  + [max(eps,p-a), min(p+a,1)] : true;
endmodule

module policy
	[up]    x < 3  -> true;
	[left]  x >= 3 -> true;
	[right] false -> true;
endmodule

label "crashed" = crashed;
label "goal" = goal;
"""
    
    if jamming is not None:
        prism_template = patch_base_for_jamming(prism_template)
        prism_template += "\n" + emit_jamming_block(jamming)

    # Save to the current working directory
    suffix = "_jam" if jamming is not None else ""
    filename = f"uav_random_seed_{seed}{suffix}.prism"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prism_template)

    return filename


def generate_map_from_model(prism_filepath):
    """
    Parses a given PRISM file and generates a matplotlib visualization
    by extracting state valuations directly from the built model.
    """
    # Parse and build the model
    prism_program = stormpy.parse_prism_program(prism_filepath)
    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    model = stormpy.build_sparse_interval_model_with_options(prism_program, options)
    
    # Access the labels and state valuations (variable values)
    labels = model.labeling
    val = model.state_valuations

    # Resolve variables once, then use them to read per-state valuations.
    x_var = next(v for v in prism_program.variables if v.name == "x")
    y_var = next(v for v in prism_program.variables if v.name == "y")
    
    # Variables to hold our dynamically extracted map data
    max_x, max_y = 0, 0
    obstacles = []
    goal_pos = None
    start_pos = None

    # Iterate through every state in the MDP to build our map data
    for state in model.states:
        s_id = state.id
        
        # Extract the x and y coordinate for this specific state
        x = int(val.get_value(s_id, x_var))
        y = int(val.get_value(s_id, y_var))
        
        # Track the maximum dimensions to size our grid
        max_x = max(max_x, x)
        max_y = max(max_y, y)

        state_labels = labels.get_labels_of_state(s_id)
        
        if "crashed" in state_labels and (x, y) not in obstacles:
            obstacles.append((x, y))
        if "goal" in state_labels:
            goal_pos = (x, y)
        if "init" in state_labels:
            start_pos = (x, y)

    _, ax = plt.subplots(figsize=(6, 7))
    
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            facecolor = 'white'
            
            if (x, y) == goal_pos:
                facecolor = '#d9ead3' # Light green
            elif (x, y) in obstacles:
                facecolor = '#f4cccc' # Light red
                
            rect = patches.Rectangle((x - 1, y - 1), 1, 1, linewidth=1, 
                                     edgecolor='black', facecolor=facecolor)
            ax.add_patch(rect)
            
    # Draw the drone at the initial position
    if start_pos:
        ax.plot(start_pos[0] - 0.5, start_pos[1] - 0.5, marker='o', 
                markersize=20, color="#212121", zorder=3)
                
    # Configure aesthetics to match the grid
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xticks(range(max_x + 1))
    ax.set_yticks(range(max_y + 1))
    ax.set_xticklabels([str(i) for i in range(1, max_x + 2)])
    ax.set_yticklabels([str(i) for i in range(1, max_y + 2)])
    ax.grid(False)
    ax.set_aspect('equal')
    
    # Display the seed name in the title for easy tracking
    plt.title(f"Procedural Model: {os.path.basename(prism_filepath)}")
    
    plt.show()

# ==========================================
# Script Execution
# ==========================================
if __name__ == "__main__":
    # Generate a random seed
    my_seed = random.randint(1000, 9999)
    print(f"Generating environment with seed: {my_seed}...")
    
    # Create the new dynamic PRISM file
    new_prism_file = generate_random_prism(M=5, N=6, num_obstacles=3, seed=my_seed)
    print(f"Successfully saved to: {new_prism_file}")
    
    # Build the model and visualize it
    generate_map_from_model(new_prism_file)