import stormpy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

path = "/home/glawrence/Downloads/uav.prism"

def generate_map_from_model(prism_filepath):
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
    plt.title("Map Extracted Directly from Stormpy Model")
    
    plt.show()

# Run the function
generate_map_from_model(path)
