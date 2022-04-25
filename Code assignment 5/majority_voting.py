import numpy as np
import matplotlib.pyplot as plt

COMPETENCIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def compare_groups():
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    experts = []
    doctors = []
    students = []

    for c in range(1, 201):
        experts.append(calculate_probability(c, 0.85))
        doctors.append(calculate_probability(c, 0.8))
        students.append(calculate_probability(c, 0.6))

    plt.plot(experts, label='Experts', color=colors[0])
    avg_experts = [experts[0]]
    avg_experts.extend(moving_average(experts, 2)[1:])
    plt.plot(avg_experts, '--', color=colors[0], alpha=0.5)
    plt.plot(doctors, label='Doctors', color=colors[1])
    avg_doctors = [doctors[0]]
    avg_doctors.extend(moving_average(doctors, 2)[1:])
    plt.plot(avg_doctors, '--', color=colors[1], alpha=0.5)
    plt.plot(students, label='Students', color=colors[2])
    avg_students = [students[0]]
    avg_students.extend(moving_average(students, 2)[1:])
    plt.plot(avg_students, '--', color=colors[2], alpha=0.5)

    plt.title('Probability of a correct majority vote per\nnumber of judges for different groups')
    plt.xlabel('# judges')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


def plot_results(probabilities):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in range(len(COMPETENCIES)):
        plt.plot(probabilities[i], color=colors[i], label=f'p={COMPETENCIES[i]}')
        plt.plot(moving_average(probabilities[i], 2), '--', color=colors[i], alpha=0.5)

    plt.title('Probability of a correct majority vote per number\nof doctors for different competency levels (p)')
    plt.xlabel('# doctors')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)

    return np.convolve(interval, window, 'same')


def probability_grid():
    doctors = range(1, 101)

    probabilities = [[0] * len(doctors) for _ in COMPETENCIES]

    for x, p in enumerate(COMPETENCIES):
        for y, c in enumerate(doctors):
            probabilities[x][y] = calculate_probability(c, p)

    return probabilities


def calculate_probability(c, p):
    probability = 0

    for x in range(np.math.ceil((c + 1) / 2), c + 1):
        probability += np.math.factorial(c) / (np.math.factorial(c - x) * np.math.factorial(x)) \
                       * (p ** x) * ((1 - p) ** (c - x))

    return probability


if __name__ == '__main__':
    compare_groups()
