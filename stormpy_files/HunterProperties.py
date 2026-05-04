import random
import stormpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

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
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prism_template)
        
    return filename


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
    task = stormpy.CheckTask(properties[0].raw_formula)
    task.set_produce_schedulers(True)
    task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)
    
    # Model check and extract the scheduler
    result = stormpy.check_interval_mdp(model, task, stormpy.Environment())
    scheduler = result.scheduler
    
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
    _, ax = plt.subplots(figsize=(6, 7))
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

        choice = scheduler.get_choice(state)
        if choice.defined:
            action_index = choice.get_deterministic_choice()
            
            # Map choice index to the text label ('up', 'left', 'right')
            action_name = None
            for label in model.choice_labeling.get_labels():
                if model.choice_labeling.get_choices(label).get(action_index):
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


def compute_bounded_reachability_curve(prism_filepath, max_k=30):
    """
    Interval MDP from the standard PRISM file (no horizon augmentation).
    Robust value iteration for the same semantics as the primary curve in
    plot_bounded_reachability (Pmax for !crashed U<=K goal style).

    Returns list of length (max_k + 1): entry j is the value at the initial state
    with step bound j (j = 0..max_k).
    """
    prism_program = stormpy.parse_prism_program(prism_filepath)
    options = stormpy.BuilderOptions()
    model = stormpy.build_sparse_interval_model_with_options(prism_program, options)
    initial_state = model.initial_states[0]

    labels = model.labeling
    goal_states = set()
    crashed_states = set()
    for state in model.states:
        s_labels = labels.get_labels_of_state(state.id)
        if "goal" in s_labels:
            goal_states.add(state.id)
        if "crashed" in s_labels:
            crashed_states.add(state.id)

    def robust_min_expected(action, previous_values):
        bounded = []
        lower_sum = 0.0

        for transition in action.transitions:
            interval = transition.value()
            lower = float(interval.lower())
            upper = float(interval.upper())
            target = transition.column
            bounded.append([previous_values[target], lower, upper])
            lower_sum += lower

        expected = sum(value * lower for value, lower, _ in bounded)
        remaining = max(0.0, 1.0 - lower_sum)

        bounded.sort(key=lambda x: x[0])
        for value, lower, upper in bounded:
            if remaining <= 0.0:
                break
            add = min(upper - lower, remaining)
            expected += value * add
            remaining -= add

        return expected

    pmax_u_values = []
    pmax_f_values = []
    prev_u = {}
    prev_f = {}
    for state in model.states:
        prev_u[state.id] = 1.0 if state.id in goal_states else 0.0
        prev_f[state.id] = 1.0 if state.id in goal_states else 0.0

    pmax_u_values.append(prev_u[initial_state])
    pmax_f_values.append(prev_f[initial_state])

    for _k in range(1, max_k + 1):
        curr_u = {}
        curr_f = {}

        for state in model.states:
            s_id = state.id

            if s_id in goal_states:
                curr_u[s_id] = 1.0
                curr_f[s_id] = 1.0
                continue

            if len(state.actions) == 0:
                curr_u[s_id] = 0.0
                curr_f[s_id] = 0.0
                continue

            best_f = max(robust_min_expected(action, prev_f) for action in state.actions)
            curr_f[s_id] = best_f

            if s_id in crashed_states:
                curr_u[s_id] = 0.0
            else:
                best_u = max(robust_min_expected(action, prev_u) for action in state.actions)
                curr_u[s_id] = best_u

        prev_u = curr_u
        prev_f = curr_f
        pmax_u_values.append(prev_u[initial_state])
        pmax_f_values.append(prev_f[initial_state])

    return pmax_u_values


def compute_bounded_safety_curve(prism_filepath, max_k=30):
    """
    Finite-horizon prefixes of the same robust safety iteration as plot_global_safety
    (maximize probability of never visiting 'crashed', Pmax=? [ G !"crashed" ]).

    Returns list of length (max_k + 1): entry j is the value at the initial state
    after j Bellman updates, j = 0..max_k (same indexing as compute_bounded_reachability_curve).
    """
    prism_program = stormpy.parse_prism_program(prism_filepath)
    options = stormpy.BuilderOptions()
    model = stormpy.build_sparse_interval_model_with_options(prism_program, options)
    initial_state = model.initial_states[0]
    labels = model.labeling
    crashed_states = set()
    for state in model.states:
        if "crashed" in labels.get_labels_of_state(state.id):
            crashed_states.add(state.id)

    def robust_min_expected(action, previous_values):
        bounded = []
        lower_sum = 0.0

        for transition in action.transitions:
            interval = transition.value()
            lower = float(interval.lower())
            upper = float(interval.upper())
            target = transition.column
            bounded.append([previous_values[target], lower, upper])
            lower_sum += lower

        expected = sum(value * lower for value, lower, _ in bounded)
        remaining = max(0.0, 1.0 - lower_sum)

        bounded.sort(key=lambda x: x[0])
        for value, lower, upper in bounded:
            if remaining <= 0.0:
                break
            add = min(upper - lower, remaining)
            expected += value * add
            remaining -= add

        return expected

    values = {}
    for state in model.states:
        values[state.id] = 0.0 if state.id in crashed_states else 1.0

    curve = [values[initial_state]]

    for _ in range(max_k):
        next_values = {}
        for state in model.states:
            s_id = state.id
            if s_id in crashed_states:
                next_values[s_id] = 0.0
                continue
            if len(state.actions) == 0:
                next_values[s_id] = 0.0
                continue
            next_values[s_id] = max(
                robust_min_expected(action, values) for action in state.actions
            )
        values = next_values
        curve.append(values[initial_state])

    return curve


def plot_bounded_reachability(prism_filepath, max_k=30):
    """
    Parses the PRISM file, iteratively checks bounded properties for K in [0, max_k],
    and plots the probabilities as a line graph.
    """
    print(f"Parsing model from: {prism_filepath} for bounded model checking...")
    print(f"Checking bounded properties for K up to {max_k}...")
    pmax_u_values = compute_bounded_reachability_curve(prism_filepath, max_k)
    k_values = list(range(len(pmax_u_values)))
        
    # --- Plotting the Results ---
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    # Plot both lines
    plt.plot(k_values, pmax_u_values, label='Pmax=? [ !"crashed" U<=K "goal" ]', 
             marker='o', linestyle='-', color='#1f77b4', markersize=5)
    
    # Formatting the graph
    plt.xlabel('Bound limit (K steps)', fontsize=12)
    plt.ylabel('Maximum Probability', fontsize=12)
    plt.title(f'Bounded Reachability over {max_k} Steps', fontsize=14)
    plt.xticks(range(0, max_k + 1, 2))  # Tick every 2 steps for readability
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_global_safety(prism_filepath, max_iters=200, tol=1e-8):
    """
    Computes and plots Pmax=? [ G !"crashed" ] via robust fixed-point iteration.
    """
    print(f"Parsing model from: {prism_filepath} for global safety...")
    prism_program = stormpy.parse_prism_program(prism_filepath)

    options = stormpy.BuilderOptions()
    print("Building interval MDP state space...")
    model = stormpy.build_sparse_interval_model_with_options(prism_program, options)

    initial_state = model.initial_states[0]
    labels = model.labeling
    crashed_states = set()
    for state in model.states:
        if "crashed" in labels.get_labels_of_state(state.id):
            crashed_states.add(state.id)

    def robust_min_expected(action, previous_values):
        bounded = []
        lower_sum = 0.0

        for transition in action.transitions:
            interval = transition.value()
            lower = float(interval.lower())
            upper = float(interval.upper())
            target = transition.column
            bounded.append([previous_values[target], lower, upper])
            lower_sum += lower

        expected = sum(value * lower for value, lower, _ in bounded)
        remaining = max(0.0, 1.0 - lower_sum)

        bounded.sort(key=lambda x: x[0])
        for value, lower, upper in bounded:
            if remaining <= 0.0:
                break
            add = min(upper - lower, remaining)
            expected += value * add
            remaining -= add

        return expected

    print(f"Computing global safety fixed point (max_iters={max_iters}, tol={tol})...")
    values = {}
    for state in model.states:
        values[state.id] = 0.0 if state.id in crashed_states else 1.0

    iteration_values = [values[initial_state]]

    for _ in range(max_iters):
        next_values = {}
        max_diff = 0.0

        for state in model.states:
            s_id = state.id
            if s_id in crashed_states:
                next_values[s_id] = 0.0
                continue

            if len(state.actions) == 0:
                next_values[s_id] = 0.0
                continue

            next_values[s_id] = max(
                robust_min_expected(action, values) for action in state.actions
            )
            max_diff = max(max_diff, abs(next_values[s_id] - values[s_id]))

        values = next_values
        iteration_values.append(values[initial_state])

        if max_diff < tol:
            break

    print("Plotting global safety results...")
    iterations = list(range(len(iteration_values)))
    plt.figure(figsize=(10, 6))
    plt.plot(
        iterations,
        iteration_values,
        marker='o',
        linestyle='-',
        color='#2ca02c',
        markersize=4,
        label='Pmax=? [ G !"crashed" ] (initial state)'
    )
    plt.xlabel('Value iteration step', fontsize=12)
    plt.ylabel('Maximum Safety Probability', fontsize=12)
    plt.title('Global Safety Probability Convergence', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Generate a random seed
    my_seed = random.randint(1000, 9999)
    print(f"--- Starting Pipeline for Seed {my_seed} ---")
    
    # Generate the dynamic PRISM file
    new_prism_file = generate_random_prism(M=5, N=6, num_obstacles=3, seed=my_seed)
    
    # (Optional) Retain your original visualizer
    # synthesize_and_visualize(new_prism_file)
    
    # Add the new bounded reachability graph
    plot_bounded_reachability(new_prism_file, max_k=30)

    # Add separate graph for global safety property
    plot_global_safety(new_prism_file)
    
    print("--- Pipeline Complete ---")