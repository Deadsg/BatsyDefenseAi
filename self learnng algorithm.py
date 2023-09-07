def objective_function(x):
    return -(x ** 2)  # Negative because we want to find the maximum

def hill_climbing(starting_point, step_size, num_iterations):
    current_point = starting_point

    for _ in range(num_iterations):
        current_value = objective_function(current_point)

        # Evaluate neighboring points
        left_neighbor = current_point - step_size
        right_neighbor = current_point + step_size

        left_value = objective_function(left_neighbor)
        right_value = objective_function(right_neighbor)

        # Move to the neighbor with the higher value
        if left_value > current_value:
            current_point = left_neighbor
        elif right_value > current_value:
            current_point = right_neighbor

    return current_point, objective_function(current_point)

if __name__ == "__main__":
    starting_point = 2
    step_size = 0.1
    num_iterations = 100

    final_point, max_value = hill_climbing(starting_point, step_size, num_iterations)

    print(f"The maximum value is {max_value} at x = {final_point}")