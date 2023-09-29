import random, operator
import numpy as np
import pandas as pd


#Introduction to the Traveling Salesman Problem
print("Genetic Algorithm for Traveling Salesman Problem")
print("This code finds an optimal route for visiting a set of cities, minimizing the total distance traveled.")
print("The algorithm starts with a population of random routes, and then breeds them to create a new generation of routes.")
# Define a City class with functions for calculating distances.
class City:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  #computes the distance between two cities
  def distance(self, city):
    distanceX = abs(self.x - city.x)
    distanceY = abs(self.y - city.y)

    distance = np.sqrt((distanceX**2) + (distanceY**2))
    return distance

  def __repr__(self):
    return "(" + str(self.x) + "," + str(self.y) + ")"
# Define a Fitness class to track fitness and distance for a route.
class Fitness:
  # Initializes variables for each chromosome
  def __init__(self, route):
    self.route = route
    self.distance = 0
    self.fitness = 0.0

  #Calculates the total distance of a route
  def routeTotalDistance(self):
    #If the total distance of a route is uninitialized
    if self.distance == 0:
      pathDistance = 0
      # Calculate the distance between each pair of cities in the route
      for i in range(0, len(self.route)):
        fromCity = self.route[i]
        toCity = None
        if i + 1 < len(self.route):
          toCity = self.route[i + 1]
        else:
          toCity = self.route[0]
        pathDistance += fromCity.distance(toCity)
      self.distance = pathDistance
    return self.distance

  #Calculates the fitness of each chromosome (inverse of the total distance)
  def determineFitness(self):
    if(self.fitness == 0):
      self.fitness = 1/ float(self.routeTotalDistance())
    return self.fitness

# Initializes a list of 25 cities with x and y coordinates.
cityList = []
for i in range(0, 25):
  cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

#This function creates a randomized array of cities to be visited in a certain order.
def createRoute(cityList):
  route = random.sample(cityList, len(cityList))
  return route

# Initializes the population of chromosomes by shuffling the citylist in random order to create diversity.
def initializePopultaion(populationSize, cityList):
  population = []
  for i in range(0, populationSize):
    population.append(createRoute(cityList))
  return population

# Creates a dictionary, where each index of the population is mapped to its fitness.
# Than orders the indexes of each population by their fitness in order from greatest to least.
def rankRoutes(population):
  fitnessRanked = {}
  for i in range(0, len(population)):
    fitnessRanked[i] = Fitness(population[i]).determineFitness()
  return sorted(fitnessRanked.items(), key = operator.itemgetter(1), reverse = True )

#Selects the parents to breed for the next generation of chromosomes

def selection(rankedPopulation, eliteSize):
  selectionResults = []
  df = pd.DataFrame(np.array(rankedPopulation), columns=["Index","Fitness"])
  df['cum_sum'] = df.Fitness.cumsum()
  #Determines the chance that a chromosome may be selected for breeding
  df["cum_perc"] = 100 * df.cum_sum/df.Fitness.sum()

  #Retains the highest fitness members of the population
  for i in range(0, eliteSize):
    selectionResults.append(rankedPopulation[i][0])

  #generates a random number, if the fitness rank is within a certain threshold of the random number, the member of the population is chosen.
  for i in range(0, len(rankedPopulation) - eliteSize):
      pick = 100*random.random()
      for i in range(0, len(rankedPopulation)):
          if pick <= df.iat[i,3]:
              selectionResults.append(rankedPopulation[i][0])
              break
  return selectionResults

# Retrieve the actual chromosomes from their indexes.
def matingPool(population, selectionResults):
  matingPool = []
  for i in range(0, len(selectionResults)):
    index = selectionResults[i]
    matingPool.append(population[index])
  return matingPool

#Produces a child from two chromosomes
def breed(parentOne, parentTwo):
  child = []
  # These arrays receive data from each of the parent that will be combined into the new child.
  childPartOne = []
  childPartTwo = []

  #Choose a random set of information from parent One that will be passed down to the child.
  geneA = int(random.random() * len(parentOne))
  geneB = int(random.random() * len(parentOne))
  startPoint = min(geneA, geneB)
  endPoint = max(geneA, geneB)

  for i in range(startPoint, endPoint):
    childPartOne.append(parentOne[i])

  #Fill in the remaining needed data in the child with data that does not already exist in it.
  childPartTwo = [item for item in parentTwo if item not in childPartOne]
  child = childPartOne + childPartTwo
  return child

# Create the next generation by breeding.
def breedPopulation(matingPool, eliteSize):
  children = []
  
  length = len(matingPool) - eliteSize
  pool = random.sample(matingPool, len(matingPool))

  # Preserve elite chromosomes.
  for i in range(0, eliteSize):
    children.append(matingPool[i])

  #Creates partners for breeding
  for i in range(0, length):
    child = breed(pool[i], pool[len(matingPool) - i - 1])
    children.append(child)
  return children

#Mutates a chromosome by swapping two cities in the route.
def mutate(individual, mutationRate):
  for swapped in range(len(individual)):
    if(random.random() < mutationRate):
      swapWith = int(random.random() * len(individual))

      city1 = individual[swapped]
      city2 = individual[swapWith]

      individual[swapped] = city2
      individual[swapWith] = city1
  return individual

#Mutate each individual in the population, and passes them to the mutate function.
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#This function calls all functions needed to develop next generation.
def nextGeneration(currentGen, eliteSize, mutationRate):
  popRanked = rankRoutes(currentGen)
  selectionResults = selection(popRanked, eliteSize)
  matingpool = matingPool(currentGen, selectionResults)
  children = breedPopulation(matingpool, eliteSize)
  nextGeneration = mutatePopulation(children, mutationRate)
  return nextGeneration

#This function runs the genetic algorithm for a given amount of generations.
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
  #Creates initial population
  pop = initializePopultaion(popSize, population)
  print("Initial solution: " + str(1 / rankRoutes(pop)[0][1]))

  previousBest = (1 / rankRoutes(pop)[0][1])
  print(previousBest)

  #This is a variable that checks if the algorithm has barely increased its solution in, if it has not by a certain amount of runs, stop running.
  runEnder = 0

  totalGenerations = 0
  #Runs the algorithm for a given amount of generations
  for i in range(0, generations):
    if runEnder >= 500:
      break
    totalGenerations += 1
    pop = nextGeneration(pop, eliteSize, mutationRate)
    currentBest = (1 / rankRoutes(pop)[0][1])
    dif = abs(currentBest - previousBest)
    if(dif < 0.00000000000000000001):
      runEnder += 1
    else:
      previousBest = currentBest
    # Print the progress for each generation
    # Print the best sequence of cities and their distances
    best_route_index = rankRoutes(pop)[0][0]
    best_route = pop[best_route_index]
    print(f"Generation {i+1}: Best Distance = {currentBest},\n Best Route = {best_route}")

  print("Optimized solution: " + str(1 / rankRoutes(pop)[0][1]))
  print("Total generations: " + str(totalGenerations))
  bestRouteIndex = rankRoutes(pop)[0][0]
  bestRoute = pop[bestRouteIndex]
  return bestRoute

geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500)
