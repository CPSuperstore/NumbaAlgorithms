import math
import typing

import numba
import numpy as np


@numba.jit(nopython=True)
def calculate_weights(parent_population_size: int):
    weights = []

    for i in range(parent_population_size):
        numerator = math.log10(parent_population_size + 0.5) - math.log10(i + 1)

        denominator = 0
        for j in range(parent_population_size):
            denominator += math.log10(parent_population_size + 0.5) - math.log10(j + 1)

        weights.append(numerator / denominator)

    return np.array(weights)


@numba.jit(nopython=True)
def canonical_es(
        environment: callable, generations: int, offspring_population_size: int,
        policy_parameters: typing.Union[np.ndarray, list], parent_population_percent: float = 0.25,
        mutation_step_size: float = 1, maximize: bool = True, show_every: int = 1000
):
    parent_population_size = int(round(parent_population_percent * offspring_population_size))
    policy_parameter_count = len(policy_parameters)

    weights = calculate_weights(parent_population_size)

    for generation in range(generations):
        scores = np.zeros(offspring_population_size).astype("float64")
        noises = np.zeros((offspring_population_size, policy_parameter_count)).astype("float64")

        for child in range(offspring_population_size):
            noise = np.random.standard_normal(policy_parameter_count)
            noises[child, ] = noise
            scores[child] = environment(policy_parameters + mutation_step_size * noise)

        if maximize:
            indices = scores.argsort()[-parent_population_size:][::-1]

        else:
            indices = scores.argsort()[::-1][-parent_population_size:][::-1]

        new_weights = noises[indices] * weights.reshape((len(weights), 1))

        averaged_weights = np.zeros(policy_parameter_count)
        for i in range(policy_parameter_count):
            averaged_weights[i] = np.mean(new_weights[:, i])

        policy_parameters += mutation_step_size * averaged_weights

        if (generation + 1) % show_every == 0:
            print("Generation", generation + 1)

    return policy_parameters
