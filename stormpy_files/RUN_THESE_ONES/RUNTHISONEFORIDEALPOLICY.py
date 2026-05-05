import os
import random
from collections import deque

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import stormpy


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

		max_w = min(6, max(2, M - 1))
		max_h = min(6, max(2, N - 1))

		if max_w < 2 or max_h < 2:
			raise ValueError("Grid is too small to place a building")

		for _ in range(1000):
			if random.random() < 0.5 and max_w >= 3:
				w = random.randint(3, max_w)
				h = 2
			else:
				w = 2
				h = random.randint(3, max_h)

			x = random.randint(2, M - w + 1)
			y = random.randint(2, N - h + 1)

			candidate = make_building(M, N, x, y, w, h)

			if is_hospital and any(in_inner_region(cx, cy, M, N) for cx, cy in candidate):
				continue

			if can_place(candidate):
				buildings.append(candidate)
				occupied.update(candidate)

				if is_hospital:
					hospital = candidate

				return True

		return False

	place_building(is_hospital=True)

	for _ in range(num_buildings):
		place_building()

	building_cells = set().union(*buildings) if buildings else set()
	hospital_cells = hospital if hospital is not None else set()

	all_cells = {
		(x, y)
		for x in range(1, M + 1)
		for y in range(1, N + 1)
	}

	free_spaces = all_cells - (building_cells | hospital_cells)

	return building_cells, hospital_cells, free_spaces


def generate_random_prism(M: int = 5, N: int = 6, num_obstacles: int = 3, seed=None):
	"""Generate a PRISM file with a randomized city, hospital goal, and obstacles."""
	if seed is not None:
		random.seed(seed)

	buildings, hospital, free_spaces = generate_city_with_hospital(
		M=M, N=N, num_buildings=num_obstacles, seed=seed
	)

	obstacle_positions = sorted(buildings - hospital)
	goal_positions = sorted(hospital)

	def far_enough(x, y, targets, min_dist=10):
		for tx, ty in targets:
			if abs(x - tx) + abs(y - ty) < min_dist:
				return False
		return True

	def sample_start(free_spaces, hospital_cells, min_dist=10):
		free_list = list(free_spaces)

		for _ in range(10000):
			x, y = random.choice(free_list)
			if far_enough(x, y, hospital_cells, min_dist):
				return x, y

		raise ValueError("Could not find valid start position")

	start_x, start_y = sample_start(free_spaces, hospital, min_dist=10)

	goal_strings = [f"(x = {gx} & y = {gy})" for gx, gy in goal_positions]
	goal_formula = f"formula goal = {' | '.join(goal_strings)};"
	obs_strings = [f"(x = {ox} & y = {oy})" for ox, oy in obstacle_positions]
	crashed_formula = f"formula crashed = {' | '.join(obs_strings)};"

	prism_template = f"""mdp

// Model parameters
const double p = 0.01;
const double e = 0.01;
formula a = e;

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

label "crashed" = crashed;
label "goal" = goal;
"""

	filename = f"uav_random_seed_{seed}.prism"
	with open(filename, "w", encoding="utf-8") as prism_file:
		prism_file.write(prism_template)

	return filename


def synthesize_and_visualize(prism_filepath):
	"""Parse the PRISM file, synthesize a policy, and visualize the result."""
	print(f"Parsing model from: {prism_filepath}...")
	prism_program = stormpy.parse_prism_program(prism_filepath)

	options = stormpy.BuilderOptions()
	options.set_build_state_valuations()
	options.set_build_choice_labels()

	print("Building interval MDP state space...")
	model = stormpy.build_sparse_interval_model_with_options(prism_program, options)

	print('Synthesizing optimal policy (Pmax=? [ !"crashed" U "goal" ])...')
	formula_str = 'Pmax=? [ !"crashed" U "goal" ]'
	properties = stormpy.parse_properties(formula_str, prism_program)
	task = stormpy.CheckTask(properties[0].raw_formula)
	task.set_produce_schedulers(True)
	task.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.ROBUST)

	result = stormpy.check_interval_mdp(model, task, stormpy.Environment())
	scheduler = result.scheduler

	labels = model.labeling
	val = model.state_valuations
	x_var = next(v for v in prism_program.variables if v.name == "x")
	y_var = next(v for v in prism_program.variables if v.name == "y")

	max_x, max_y = 0, 0
	obstacles, goal_positions, start_pos = [], [], None

	for state in model.states:
		s_id = state.id
		x = int(val.get_value(s_id, x_var))
		y = int(val.get_value(s_id, y_var))
		max_x = max(max_x, x)
		max_y = max(max_y, y)

		state_labels = labels.get_labels_of_state(s_id)
		if "crashed" in state_labels and (x, y) not in obstacles:
			obstacles.append((x, y))
		if "goal" in state_labels and (x, y) not in goal_positions:
			goal_positions.append((x, y))
		if "init" in state_labels:
			start_pos = (x, y)

	if not goal_positions:
		raise ValueError("No goal states were found in the generated model")

	coord_to_state = {}
	state_to_coord = {}
	for state in model.states:
		s_id = state.id
		xy = (int(val.get_value(s_id, x_var)), int(val.get_value(s_id, y_var)))
		coord_to_state[xy] = state
		state_to_coord[s_id] = xy

	dist_to_goal = {coord: float("inf") for coord in coord_to_state}
	queue = deque()
	for goal in goal_positions:
		if goal in dist_to_goal:
			dist_to_goal[goal] = 0
			queue.append(goal)

	while queue:
		x, y = queue.popleft()
		for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
			nx, ny = x + dx, y + dy
			if (nx, ny) in coord_to_state and (nx, ny) not in obstacles and dist_to_goal[(nx, ny)] == float("inf"):
				dist_to_goal[(nx, ny)] = dist_to_goal[(x, y)] + 1
				queue.append((nx, ny))

	def nearest_goal(coord):
		return min(goal_positions, key=lambda goal: (abs(goal[0] - coord[0]) + abs(goal[1] - coord[1]), goal[0], goal[1]))

	def action_name_from_choice(state, local_choice_index):
		actions = list(state.actions)
		if local_choice_index < 0 or local_choice_index >= len(actions):
			return None

		selected_action = actions[local_choice_index]
		if hasattr(selected_action, "labels"):
			action_labels = list(selected_action.labels)
			if action_labels:
				return action_labels[0]

		return None

	greedy_policy = {}
	for state in model.states:
		s_id = state.id
		x, y = state_to_coord[s_id]
		state_labels = labels.get_labels_of_state(s_id)

		if "crashed" in state_labels or "goal" in state_labels:
			greedy_policy[s_id] = None
			continue

		current_dist = dist_to_goal[(x, y)]
		goal_x, goal_y = nearest_goal((x, y))
		delta_x = goal_x - x
		delta_y = goal_y - y
		preferred_axis = "x" if abs(delta_x) >= abs(delta_y) else "y"
		best_action = None
		best_score = (current_dist, 2, 2)

		actions = list(state.actions)
		for action_idx, action in enumerate(actions):
			action_labels = list(action.labels) if hasattr(action, "labels") else []
			if not action_labels:
				continue

			action_name = action_labels[0]
			axis_penalty = 2
			if action_name == "left":
				action_axis = "x"
				action_delta = -1
			elif action_name == "right":
				action_axis = "x"
				action_delta = 1
			elif action_name == "down":
				action_axis = "y"
				action_delta = -1
			elif action_name == "up":
				action_axis = "y"
				action_delta = 1
			else:
				action_axis = None
				action_delta = 0

			if action_axis == preferred_axis:
				if action_axis == "x" and ((delta_x > 0 and action_delta > 0) or (delta_x < 0 and action_delta < 0)):
					axis_penalty = 0
				elif action_axis == "y" and ((delta_y > 0 and action_delta > 0) or (delta_y < 0 and action_delta < 0)):
					axis_penalty = 0
				else:
					axis_penalty = 1
			elif action_axis is not None:
				if action_axis == "x" and ((delta_x > 0 and action_delta > 0) or (delta_x < 0 and action_delta < 0)):
					axis_penalty = 1
				elif action_axis == "y" and ((delta_y > 0 and action_delta > 0) or (delta_y < 0 and action_delta < 0)):
					axis_penalty = 1
			successor_coords = set()
			for transition in action.transitions:
				successor_coords.add(state_to_coord[transition.column])

			# For a greedy choice, we want to look at the expected behavior or Intended behavior.
			# Let's consider the intended next state for distance computation (greedy)
			nx, ny = x, y
			if action_name == "up":
				nx, ny = x, min(y+1, max_y)
			elif action_name == "down":
				nx, ny = x, max(y-1, 1)
			elif action_name == "right":
				nx, ny = min(x+1, max_x), y
			elif action_name == "left":
				nx, ny = max(x-1, 1), y

			min_successor_dist = dist_to_goal.get((nx, ny), float("inf"))

			candidate_score = (min_successor_dist, axis_penalty, 0 if action_axis == preferred_axis else 1)
			if candidate_score < best_score:
				best_score = candidate_score
				best_action = (action_idx, action_name)

		if best_action is None and actions:
			for action_idx, action in enumerate(actions):
				action_labels = list(action.labels) if hasattr(action, "labels") else []
				if action_labels:
					best_action = (action_idx, action_labels[0])
					break

		greedy_policy[s_id] = best_action

	_, ax = plt.subplots(figsize=(8, 8))
	for x in range(1, max_x + 1):
		for y in range(1, max_y + 1):
			facecolor = "white"
			if (x, y) in goal_positions:
				facecolor = "#2ca02c"
			elif (x, y) in obstacles:
				facecolor = "#d62728"

			rect = patches.Rectangle(
				(x - 1, y - 1),
				1,
				1,
				linewidth=1,
				edgecolor="black",
				facecolor=facecolor,
			)
			ax.add_patch(rect)

	if start_pos:
		ax.plot(
			start_pos[0] - 0.5,
			start_pos[1] - 0.5,
			marker="o",
			markersize=20,
			color="#212121",
			zorder=3,
			alpha=0.3,
		)

	print("Overlaying greedy guidance policy arrows onto the map...")
	action_counts = {"up": 0, "down": 0, "left": 0, "right": 0}
	for state in model.states:
		s_id = state.id
		x, y = state_to_coord[s_id]
		state_labels = labels.get_labels_of_state(s_id)

		if "crashed" in state_labels or "goal" in state_labels:
			continue

		greedy_choice = greedy_policy.get(s_id)
		if greedy_choice is None:
			continue

		action_idx, action_name = greedy_choice
		action_name = action_name or action_name_from_choice(state, action_idx)

		if action_name in action_counts:
			action_counts[action_name] += 1

		dx, dy = 0, 0
		arrow_length = 0.35
		if action_name == "up":
			dy = arrow_length
		elif action_name == "down":
			dy = -arrow_length
		elif action_name == "right":
			dx = arrow_length
		elif action_name == "left":
			dx = -arrow_length

		if dx != 0 or dy != 0:
			ax.arrow(
				x - 0.5,
				y - 0.5,
				dx,
				dy,
				head_width=0.15,
				head_length=0.15,
				fc="#1f77b4",
				ec="#1f77b4",
				length_includes_head=True,
				zorder=4,
			)

	print(f"Selected actions in greedy guidance policy: {action_counts}")

	ax.set_xlim(0, max_x)
	ax.set_ylim(0, max_y)
	ax.set_xticks(range(max_x + 1))
	ax.set_yticks(range(max_y + 1))
	ax.set_xticklabels([str(i) for i in range(1, max_x + 2)])
	ax.set_yticklabels([str(i) for i in range(1, max_y + 2)])
	ax.grid(False)
	ax.set_aspect("equal")
	plt.title(f"Synthesized UAV Policy (Seed: {os.path.basename(prism_filepath).split('_')[-1].split('.')[0]})")
	plt.show()
	plt.savefig("output.png")
	plt.close()


if __name__ == "__main__":
	my_seed = 1596#random.randint(1000, 9999)
	print(f"Generating environment with seed: {my_seed}...")

	M, N = 30, 30
	new_prism_file = generate_random_prism(M=M, N=N, num_obstacles=30, seed=my_seed)
	print(f"Successfully saved to: {new_prism_file}")

	synthesize_and_visualize(new_prism_file)
