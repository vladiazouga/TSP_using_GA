import random
import numpy as np
import pandas as pd

# Initial print statements explaining the purpose of the code.
print("Genetic Algorithm for Traveling Salesman Problem")
print("------------------------------------------------")
print("This code demonstrates a genetic algorithm to solve the Traveling Salesman Problem (TSP).")
print("The TSP involves finding the shortest possible route that visits a set of cities once and returns to the starting city.")
print("In this implementation:")
print("- A population of routes is evolved over generations.")
print("- The fitness of each route is determined by its total distance.")
print("- Routes with shorter distances are more fit and have a higher chance of being selected for reproduction.")
print("- Crossover and mutation operations are used to create new routes in each generation.")
print("- The algorithm continues until a stopping criterion is met.")
print("Let's optimize a route for visiting a set of randomly generated cities.")
print("\nOptimizing route...\n")

# Define a City class to represent cities with x and y coordinates.
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to(self, other_city):
        distance_x = abs(self.x - other_city.x)
        distance_y = abs(self.y - other_city.y)
        return np.sqrt(distance_x**2 + distance_y**2)

    def __repr__(self):
        return f"({self.x},{self.y})"

# Define a Fitness class to calculate the fitness (inverse of distance) of a route.
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def calculate_route_distance(self):
        if self.distance == 0:
            path_distance = sum(self.route[i].distance_to(self.route[i + 1]) for i in range(len(self.route) - 1))
            path_distance += self.route[-1].distance_to(self.route[0])
            self.distance = path_distance
        return self.distance

    def calculate_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / self.calculate_route_distance()
        return self.fitness

# Function to create a random route by shuffling a list of cities.
def create_random_route(city_list):
    return random.sample(city_list, len(city_list))

# Function to initialize a population of random routes.
def initialize_population(population_size, city_list):
    return [create_random_route(city_list) for _ in range(population_size)]

# Function to rank routes in a population based on fitness.
def rank_routes(population):
    return sorted(enumerate(Fitness(route).calculate_fitness() for route in population), key=lambda x: x[1], reverse=True)

# Function to select the elite routes from the ranked population.
def select_elite(ranked_population, elite_size):
    selected_indexes = [index for index, _ in ranked_population[:elite_size]]
    return selected_indexes

# Function to create a mating pool from the selected elite routes.
def create_mating_pool(population, selected_indexes):
    return [population[index] for index in selected_indexes]

# Function to perform crossover (reproduction) between two parent routes.
def crossover(parent_one, parent_two):
    gene_a, gene_b = sorted(random.sample(range(len(parent_one)), 2))
    child_part_one = parent_one[gene_a:gene_b]
    child_part_two = [city for city in parent_two if city not in child_part_one]
    return child_part_one + child_part_two

# Function to breed a new population from the mating pool.
def breed_population(mating_pool, elite_size):
    elite = mating_pool[:elite_size]
    non_elite = mating_pool[elite_size:]
    children = [crossover(random.choice(elite), random.choice(elite)) for _ in range(len(non_elite))]
    return elite + children

# Function to perform mutation on a route with a given mutation rate.
def mutate_route(route, mutation_rate):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]
    return route

# Function to perform mutation on a population.
def mutate_population(population, mutation_rate):
    return [mutate_route(route, mutation_rate) for route in population]

# Function to evolve the current generation to the next generation.
def evolve_to_next_generation(current_generation, elite_size, mutation_rate):
    ranked_population = rank_routes(current_generation)
    selected_indexes = select_elite(ranked_population, elite_size)
    mating_pool_result = create_mating_pool(current_generation, selected_indexes)
    children = breed_population(mating_pool_result, elite_size)
    next_generation = mutate_population(children, mutation_rate)
    return next_generation

# Main genetic algorithm function to find the optimized route.
def genetic_algorithm(city_list, population_size=None, elite_size=None, mutation_rate=None, generations=None):
        # Prompt for the number of cities
    num_cities = int(input("Enter the number of cities: "))

    # Generate a list of random cities based on the user-defined number
    city_list = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(num_cities)]

    # Print the names of the cities in the tour and its total distance
    print("Number of cities in the tour:", len(city_list))

    if population_size is None:
        pop_size = int(input("Enter the integer population size (default = 100): "))
    else:
        pop_size = population_size

    if mutation_rate is None:
        mutation_rate = float(input("Enter the float mutation rate (default = 0.01): "))

    if elite_size is None:
        elite_size = int(input("Enter the proportion of elite routes in each generation (default = 20): "))

    if generations is None:
        generations = int(input("Enter the number of generations (default = 500): "))

    population = initialize_population(pop_size, city_list)
    print(f"Initial solution: {1 / rank_routes(population)[0][1]}")

    previous_best = 1 / rank_routes(population)[0][1]
    run_ender = 0
    total_generations = 0

    for _ in range(generations):
        if run_ender >= 100:
            break
        total_generations += 1
        population = evolve_to_next_generation(population, elite_size, mutation_rate)
        current_best = 1 / rank_routes(population)[0][1]
        dif = abs(current_best - previous_best)
        if dif < 1e-20:
            run_ender += 1
        else:
            previous_best = current_best

        # Print the best distance in each generation.
        print(f"Generation {total_generations}: Best Distance = {1 / current_best}")

    print(f"Optimized solution: {1 / rank_routes(population)[0][1]}")
    print(f"Total generations: {total_generations}")
    best_route_index = rank_routes(population)[0][0]
    best_route = population[best_route_index]
    print(f"Best Route: {best_route}")
    return best_route

# Additional code to gather input parameters
print("----------------------------------------------------------------")
print("Parameters for the genetic algorithm")
print("----------------------------------------------------------------")

# Print the names of the cities in the best tour and its total distance
print("Number of cities in the tour:", len(city_list))

# Prompt for population size
pop_size = int(input("Enter the integer population size (default = 100): "))

# Prompt for mutation rate
mutation_rate = float(input("Enter the float mutation rate (default = 0.01): "))

# Prompt for elite size
elite_size = int(input("Enter the proportion of elite routes in each generation (default = 20): "))

# Prompt for the number of generations
generations = int(input("Enter the number of generations (default = 500): "))

print("Number of stagnant generations: 100")
print("\n")

# Run the genetic algorithm with user-defined or default parameters
genetic_algorithm(city_list, population_size=pop_size, elite_size=elite_size, mutation_rate=mutation_rate, generations=generations)
