import random
import stormpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# ==========================================
# 1. Procedural Generation
# ==========================================
def generate_random_prism(M=5, N=6, num_obstacles=3, seed=None):
    """
    Generates a PRISM file with randomly placed obstacles and goal.
    Does NOT include a manual policy, leaving it open for synthesis.
    Returns the file path of the generated model.
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

    # The PRISM Template (Environment only, no manual policy)
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

label "crashed" = crashed;
label "goal" = goal;
"""
    
    filename = f"uav_env_{seed}.prism"
    with open(filename, "w") as f:
        f.write(prism_template)
        
    return filename


# ==========================================
# 2. Synthesis & Visualization
# ==========================================
def synthesize_and_visualize(prism_filepath):
    """
    Parses the PRISM file, synthesizes the optimal policy via PCTL,
    and plots both the grid and the policy vectors.
    """
    print(f"Parsing model from: {prism_filepath}...")
    prism_program = stormpy.parse_prism_program(prism_filepath)
    
    # Configure builder to retain variable coordinates and action names
    options = stormpy.BuilderOptions()
    options.set_build_state_valuations()
    options.set_build_choice_labels() 
    
    print("Building interval MDP state space...")
    model = stormpy.build_sparse_interval_model_with_options(prism_program, options)

    # Synthesize the Optimal Policy
    print("Synthesizing optimal policy (Pmax=? [ !crashed U goal ])...")
    formula_str = 'Pmax=? [ !"crashed" U "goal" ]'
    properties = stormpy.parse_properties(formula_str, prism_program)
    
    # Model check and extract the scheduler
    result = stormpy.model_checking(model, properties[0], extract_scheduler=True)
    scheduler = result.get_scheduler()
    
    # Extract State Data for mapping
    labels = model.labeling
    val = model.state_valuations
    x_var = next(v for v in prism_program.variables if v.name == "x")
    y_var = next(v for v in prism_program.variables if v.name == "y")
    
    max_x, max_y = 0, 0
    obstacles, goal_pos, start_pos = [], None, None

    for state in model.states:
        s_id = state.id
        x, y = int(val.get_value(s_id, x_var)), int(val.get_value(s_id, y_var))
        max_x, max_y = max(max_x, x), max(max_y, y)
        
        state_labels = labels.get_labels_of_state(s_id)
        if "crashed" in state_labels and (x, y) not in obstacles:
            obstacles.append((x, y))
        if "goal" in state_labels: goal_pos = (x, y)
        if "init" in state_labels: start_pos = (x, y)

    # Draw the Base Map
    fig, ax = plt.subplots(figsize=(6, 7))
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            facecolor = 'white'
            if (x, y) == goal_pos: facecolor = '#d9ead3'
            elif (x, y) in obstacles: facecolor = '#f4cccc'
                
            rect = patches.Rectangle((x - 1, y - 1), 1, 1, linewidth=1, 
                                     edgecolor='black', facecolor=facecolor)
            ax.add_patch(rect)

    if start_pos:
        ax.plot(start_pos[0] - 0.5, start_pos[1] - 0.5, marker='o', 
                markersize=20, color="#212121", zorder=3, alpha=0.3)

    # Overlay Policy Arrows
    print("Overlaying synthesized policy vectors onto the map...")
    for state in model.states:
        s_id = state.id
        x, y = int(val.get_value(s_id, x_var)), int(val.get_value(s_id, y_var))
        state_labels = labels.get_labels_of_state(s_id)
        
        # Don't draw arrows pointing out of the goal or an obstacle
        if "crashed" in state_labels or "goal" in state_labels:
            continue

        if scheduler.is_defined(state):
            choice = scheduler.get_choice(state)
            action_index = choice.get_deterministic_choice()
            
            # Map choice index to the text label ('up', 'left', 'right')
            action_name = None
            for label in model.choice_labeling.get_labels():
                if model.choice_labeling.get_states_and_choices(label).contains(s_id, action_index):
                    action_name = label
                    break
            
            # Calculate arrow vector
            dx, dy = 0, 0
            arrow_length = 0.35
            if action_name == "up": dy = arrow_length
            elif action_name == "down": dy = -arrow_length
            elif action_name == "right": dx = arrow_length
            elif action_name == "left": dx = -arrow_length
            
            # Draw it
            if dx != 0 or dy != 0:
                ax.arrow(x - 0.5, y - 0.5, dx, dy, 
                         head_width=0.15, head_length=0.15, 
                         fc='#1f77b4', ec='#1f77b4', 
                         length_includes_head=True, zorder=4)

    # Set graphical limits and ticks
    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xticks(range(max_x + 1))
    ax.set_yticks(range(max_y + 1))
    ax.set_xticklabels([str(i) for i in range(1, max_x + 2)])
    ax.set_yticklabels([str(i) for i in range(1, max_y + 2)])
    ax.grid(False)
    ax.set_aspect('equal')
    plt.title(f"Synthesized UAV Policy (Seed: {os.path.basename(prism_filepath).split('_')[-1].split('.')[0]})")
    plt.show()


# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    # Generate a random seed
    my_seed = random.randint(1000, 9999)
    print(f"--- Starting Pipeline for Seed {my_seed} ---")
    
    # Generate the dynamic PRISM file
    new_prism_file = generate_random_prism(M=5, N=6, num_obstacles=3, seed=my_seed)
    
    # Synthesize the policy and visualize the result
    synthesize_and_visualize(new_prism_file)
    
    print("--- Pipeline Complete ---")