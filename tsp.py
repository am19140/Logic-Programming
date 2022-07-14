import numpy as np
import random as rnd
# all cities by default order
default = [0, 1, 2, 3, 4]
# all distances between cities
distances = np.array([[0, 4, 4, 7, 3],
                     [4, 0, 2, 3, 5],
                     [4, 2, 0, 2, 3],
                     [7, 3, 2, 0, 6],
                     [3, 5, 3, 6, 0]])


# fitness function accepts solution as input and calculates the total distance of the
# solution / route in the distance variable
def fitness_function(solution):
    total_distance = 0
    for i in range(len(solution)-1):
        total_distance += distances[default.index(solution[i]), default.index(solution[i+1])]
    return total_distance


# function that calculates the initial population
def create_population():
    population = []
    for i in range(40):
        random_solution = default.copy()
        rnd.shuffle(random_solution)
        population.append(random_solution)
    return np.array(population)


# selecting a parent
def pick_parent(population):
    fit_bag_evals = evaluate_fitness(population)
    a = True
    while a:
        random_num = rnd.randint(0, len(population)-1)
        pick = fit_bag_evals["fitness_weight"][random_num]
        r = rnd.random()
        if r <= pick:
            parent = fit_bag_evals["solution"][random_num]
            a = False
    return parent


def evaluate_fitness(population):
    result = {}
    fit_values = []
    solutions = []
    for solution in population:
        fit_values.append(fitness_function(solution))
        solutions.append(solution)
    result["fit_values"] = fit_values
    result["solution"] = np.array(solutions)
    min_weight = [np.max(list(result["fit_values"]))-i for i in list(result["fit_values"])]
    result["fitness_weight"] = [i/sum(min_weight) for i in min_weight]
    return result


# order 1 crossover
def crossover(parent1, parent2):
    n = len(parent1)
    # creates list child of nan elements
    child = [np.nan for i in range(n)]
    num_els = np.ceil(n * (rnd.randint(10, 90) / 100))
    str_pnt = rnd.randint(0, n - 2)
    end_pnt = n if int(str_pnt + num_els) > n else int(str_pnt + num_els)
    blockA = list(parent1[str_pnt:end_pnt])
    child[str_pnt:end_pnt] = blockA
    for i in range(n):
        if list(blockA).count(parent2[i]) == 0:
            for j in range(n):
                if np.isnan(child[j]):
                    child[j] = parent2[i]
                    break
    return child


def mutation(solution):
    n = len(solution)
    x = rnd.randint(0, n-1)
    y = rnd.randint(0, n-1)
    result = swap(solution, x, y)
    return result


def swap(mylist, position1, position2):
    mylist[position1], mylist[position2] = mylist[position2], mylist[position1]
    return mylist


pop = create_population()

for x in range(200):

    pop_fit = evaluate_fitness(pop)

    # Best individual in the current population
    best_fitness = np.min(pop_fit["fit_values"])
    best_fitness_index = pop_fit["fit_values"].index(best_fitness)
    best_solution = pop_fit["solution"][best_fitness_index]

    if x == 0:
        best_fitness_ever = best_fitness
        best_solution_ever = best_solution
    else:
        if best_fitness <= best_fitness_ever:
            best_fitness_ever = best_fitness
            best_solution_ever = best_solution

    # Create the new population bag
    new_pop = []

    for i in range(10):
        # Pick 2 parents from population
        par1 = pick_parent(pop)
        par2 = pick_parent(pop)
        child = par1
        child = crossover(par1, par2)
        if rnd.random() <= 0.7:
            child = mutation(child)
        new_pop.append(child)

    pop = np.array(new_pop)


print(f"Route: {best_solution_ever}")
print(f"Total distance of route: {best_fitness_ever}")
