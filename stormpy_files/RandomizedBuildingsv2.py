import random
import stormpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def make_building(M, N, x, y, w, h):
    return {
        (i, j)
        for i in range(x, x + w)
        for j in range(y, y + h)
        if 1 <= i <= M and 1 <= j <= N
    }


def generate_city_with_hospital(M, N, num_buildings, seed=None):
    if seed is not None:
        random.seed(seed)

    buildings = []
    hospital = None
    occupied = set()

    def can_place(candidate):
        for x, y in candidate:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if (x + dx, y + dy) in occupied:
                        return False
        return True
    
    def in_inner_region(x, y, M, N, margin=0.1):
        x_min = int(M * margin)
        x_max = int(M * (1 - margin))
        y_min = int(N * margin)
        y_max = int(N * (1 - margin))

        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def place_building(is_hospital=False):

        nonlocal hospital

        for _ in range(1000):

            if random.random() < 0.5:
                w = random.randint(3, 6)
                h = 2
            else:
                w = 2
                h = random.randint(3, 6)

            x = random.randint(2, M - w)
            y = random.randint(2, N - h)

            candidate = make_building(M, N, x, y, w, h)

            if is_hospital:
                if any(in_inner_region(cx, cy, M, N) for cx, cy in candidate):
                    continue

            if can_place(candidate):
                buildings.append(candidate)
                occupied.update(candidate)

                if is_hospital:
                    hospital = candidate

                return True

        return False

    # -------------------
    # place hospital first
    # -------------------
    place_building(is_hospital=True)

    # -------------------
    # place buildings
    # -------------------
    for _ in range(num_buildings):
        place_building()

    # -------------------
    # FLATTEN SETS
    # -------------------
    building_cells = set().union(*buildings)
    hospital_cells = hospital if hospital is not None else set()

    # -------------------
    # FREE SPACE = complement
    # -------------------
    all_cells = {
        (x, y)
        for x in range(1, M + 1)
        for y in range(1, N + 1)
    }

    free_spaces = all_cells - (building_cells | hospital_cells)

    return building_cells, hospital_cells, free_spaces



def generate_random_prism(M: int = 5, N: int = 6, num_obstacles: int = 3, seed=None):
    """
    Generates a PRISM file with randomly placed obstacles and goal.
    Returns the file path of the generated model.
    """
    if seed is not None:
        random.seed(seed)
        
    buildings, hospital, free_spaces = generate_city_with_hospital(M=M, N=N, num_buildings=num_obstacles)

    obstacle_positions = list(buildings - hospital)
    goal_positions = list(hospital)

    def far_enough(x, y, targets, min_dist=10):
        for tx, ty in targets:
            if abs(x - tx) + abs(y - ty) < min_dist:
                return False
        return True
    def sample_start(free_spaces, hospital_cells, M, N, min_dist=10):
        free_list = list(free_spaces)

        for _ in range(10000):
            x, y = random.choice(free_list)

            if far_enough(x, y, hospital_cells, min_dist):
                return x, y

        raise ValueError("Could not find valid start position")

    start_x, start_y = sample_start(free_spaces, hospital, M, N, min_dist=10)
    
    # Format the PRISM formulas dynamically
    # goal_formula = f"formula goal = x = {goal_pos[0]} & y = {goal_pos[1]};"
    goal_strings = [f"(x = {ox} & y = {oy})" for ox, oy in goal_positions]
    goal_formula = f"formula goal = {' | '.join(goal_strings)};"
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
	x : [1..M] init {start_x};
	y : [1..N] init {start_y};
	[up]    !crashed & y < N -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (y'=min(y+1,N))
			  + [max(eps,p-a), min(p+a,1)] : (y'=min(y+1,N)) & (x'=max(x-1,1))
			  + [max(eps,p-a), min(p+a,1)] : (y'=min(y+1,N)) & (x'=min(x+1,M))
			  + [max(eps,p-a), min(p+a,1)] : true;
    [down]  !crashed & y > 1 -> [max(eps,1-3*(p+a)), min(1-3*(p-a),1)] : (y'=max(y-1,1))
              + [max(eps,p-a), min(p+a,1)] : (y'=max(y-1,1)) & (x'=max(x-1,1))
              + [max(eps,p-a), min(p+a,1)] : (y'=max(y-1,1)) & (x'=min(x+1,M))
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
    
    # Save to the current working directory
    filename = f"uav_random_seed_{seed}.prism"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prism_template)
        
    return filename


def generate_map_from_model(prism_filepath, M, N):
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
    goals = []
    # goal_pos = None
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
        # max_x = M
        # max_y = N

        state_labels = labels.get_labels_of_state(s_id)
        
        if "crashed" in state_labels and (x, y) not in obstacles:
            obstacles.append((x, y))
        if "goal" in state_labels and (x, y) not in goals:
            goals.append((x,y))
            # goal_pos = (x, y)
        if "init" in state_labels:
            start_pos = (x, y)

    _, ax = plt.subplots(figsize=(6, 7))
    
    for x in range(1, max_x + 1):
        for y in range(1, max_y + 1):
            facecolor = 'white'
            
            if (x, y) in goals:
                facecolor = "#79e951" # green
            elif (x, y) in obstacles:
                facecolor = "#fd5959" # red
                
            rect = patches.Rectangle((x - 1, y - 1), 1, 1, linewidth=1, 
                                     edgecolor='black', facecolor=facecolor)
            ax.add_patch(rect)
            
    # Draw the drone at the initial position
    if start_pos:
        ax.plot(start_pos[0] - 0.5, start_pos[1] - 0.5, marker='o', 
                markersize=5, color="#212121", zorder=3)
                
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

    plt.savefig("output.png")

# ==========================================
# Script Execution
# ==========================================
if __name__ == "__main__":
    # Generate a random seed
    my_seed = random.randint(1000, 9999)
    print(f"Generating environment with seed: {my_seed}...")
    
    M, N  = 30, 30

    # Create the new dynamic PRISM file
    new_prism_file = generate_random_prism(M=M, N=N, num_obstacles=30, seed=my_seed)
    print(f"Successfully saved to: {new_prism_file}")
    
    # Build the model and visualize it
    generate_map_from_model(new_prism_file, M, N)