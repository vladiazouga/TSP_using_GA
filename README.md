# TSP_using_GA


# Traveling Salesman Problem Solver using Genetic Algorithm

This Python code is an implementation of a Genetic Algorithm designed to tackle the Traveling Salesman Problem (TSP). The TSP involves determining the shortest possible route that visits a specified set of cities exactly once and returns to the starting city.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [Code Overview](#code-overview)
- [Customization](#customization)
- [Results](#results)


## Introduction

The Traveling Salesman Problem (TSP) is a well-known optimization challenge, and this code offers a solution by leveraging a genetic algorithm. The genetic algorithm evolves a population of routes across multiple generations to find the optimal route.

## Requirements

- Python 3.x
- Required Libraries: `numpy`, `pandas`

You can install the necessary libraries using `pip`:



## How to Use

1. Clone or download this code repository to your local machine.

2. Open a terminal or command prompt and navigate to the directory containing the code.

3. Execute the script using Python:

n tsp_genetic_algorithm.py


4. The code will commence solving the TSP for a randomly generated set of cities. It will display the progress and reveal the best route that has been identified.

## Code Overview

- The code defines two classes: `City` and `Fitness`, which respectively represent cities and compute route fitness.
- It initializes a list of cities with random coordinates.
- The genetic algorithm encompasses population initialization, route ranking, selection, crossover (breeding), mutation, and evolutionary improvement across several generations.
- The algorithm's objective is to discover the shortest route by iteratively enhancing the population.

## Customization

You have the flexibility to customize the algorithm by adjusting the following parameters within the `geneticAlgorithm` function:

- `popSize`: Defines the size of the population.
- `eliteSize`: Determines the number of elite chromosomes retained in each generation.
- `mutationRate`: Sets the probability of mutation.
- `generations`: Specifies the number of generations for which the algorithm runs.

Additionally, you can modify the number of cities or their coordinates by making changes to the `cityList` during code execution.

## Results

The code will provide output indicating both the initial and optimized solutions, along with the total number of generations required to find the solution. The best route discovered will be presented, including the sequence of city visits and the total distance traveled.

