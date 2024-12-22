import random
import networkx as nx
from config.config import LicenseType


def genetic_license_distribution(
    nx_graph: nx.Graph,
    prices: tuple[float, float],
    max_group_size: int,
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.01,
):
    vertices = list(nx_graph.nodes())
    num_vertices = len(vertices)
    C1, Ck = prices

    population = [random_chromosome(num_vertices) for _ in range(population_size)]

    for gen in range(generations):
        fitness_scores = []
        for chrom in population:
            cost = chromosome_cost(chrom, nx_graph, C1, Ck, max_group_size)
            fitness_scores.append(cost)

        zipped = list(zip(population, fitness_scores))
        zipped.sort(key=lambda x: x[1])
        survivors = zipped[: population_size // 2]
        new_population = [x[0] for x in survivors]

        while len(new_population) < population_size:
            p1 = random.choice(survivors)[0]
            p2 = random.choice(survivors)[0]
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population

    best_chrom, best_cost = min(
        [
            (chrom, chromosome_cost(chrom, nx_graph, C1, Ck, max_group_size))
            for chrom in population
        ],
        key=lambda x: x[1],
    )

    licenses = {
        LicenseType.INDIVIDUAL: [],
        LicenseType.GROUP_OWNER: [],
        LicenseType.GROUP_MEMBER: [],
    }

    for i, gene in enumerate(best_chrom):
        v = vertices[i]
        if gene == 1:
            licenses[LicenseType.INDIVIDUAL].append(v)
        elif gene == 2:
            licenses[LicenseType.GROUP_OWNER].append(v)

    for i, gene in enumerate(best_chrom):
        if gene == 0:
            v = vertices[i]
            owners = [
                u for u in nx_graph.neighbors(v) if best_chrom[vertices.index(u)] == 2
            ]
            if owners:
                licenses[LicenseType.GROUP_MEMBER].append(v)
            else:
                licenses[LicenseType.INDIVIDUAL].append(v)

    return licenses


def random_chromosome(length):
    return [random.choice([0, 1, 2]) for _ in range(length)]


def chromosome_cost(chromosome, nx_graph, C1, Ck, k):
    vertices = list(nx_graph.nodes())
    cost = 0.0

    owners = []
    for i, gene in enumerate(chromosome):
        if gene == 1:
            cost += C1
        elif gene == 2:
            cost += Ck
            owners.append(i)

    penalty = 0.0

    assigned_to = {o: [] for o in owners}
    for i, gene in enumerate(chromosome):
        if gene == 0:
            neighbors = [
                idx for idx in owners if nx_graph.has_edge(vertices[i], vertices[idx])
            ]
            if not neighbors:
                penalty += 9999
            else:
                assigned_to[neighbors[0]].append(i)

    for o in owners:
        if len(assigned_to[o]) + 1 > k:
            penalty += 9999 * (len(assigned_to[o]) + 1 - k)

    return cost + penalty


def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    c1 = parent1[:point] + parent2[point:]
    c2 = parent2[:point] + parent1[point:]
    return c1, c2


def mutate(chromosome, mutation_rate):
    new_chrom = []
    for gene in chromosome:
        if random.random() < mutation_rate:
            new_chrom.append(random.choice([0, 1, 2]))
        else:
            new_chrom.append(gene)
    return new_chrom
