import matplotlib.pyplot as plt
from time import sleep
import argparse

import timeit
import random
import math
from copy import deepcopy
# from xxlimited import new
from audioop import reverse
import csv
from shapely.affinity import translate
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from geopy import distance


# Source: https://realpython.com/python-csv/
def read_csv(filename):
    with open(filename, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        rows = []
        line_count = 0
        for row in csv_reader:
            rows.append(row)
        return rows


# list_dict
def get_coord_cities(cities_list):
    cities = {}
    for e in cities_list:
        cities[e["address"]] = {"geo": [float(e["geo_lat"]), float(e["geo_lon"])]}
    return cities


def get_most_populated_cities(num, cities_list):
    cities = {}
    for e in cities_list:
        cities[e["address"]] = [int(e["population"]), e]
    sorted_cities = sorted(cities.items(), key=lambda x: x[1][0], reverse=True)
    # print(sorted_cities)
    filtered_cities = []
    for name, e in sorted_cities[:num]:
        # print(e)
        # print("---------------------------")
        filtered_cities.append(e[1])

    # print(filtered_cities)
    return filtered_cities


def get_xy_gps(cities):
    new_cities = {}
    for city_name in cities.keys():
        city = cities[city_name]
        geo = city["geo"]
        xy = translate(Point(geo))
        y, x = xy.x, xy.y
        xy = {"xy": [x, y]}
        geo = {"geo": geo}
        new_cities[city_name] = {**xy, **geo}
    return new_cities


def get_distances(cities):
    new_cities = {}
    for city_name in cities.keys():
        city = cities[city_name]
        # m = city["xy"]
        m = city["geo"]
        distances = {}
        for city_name2 in cities.keys():
            city2 = cities[city_name2]
            # m2 = city2["xy"]
            m2 = city2["geo"]
            if (city_name == city_name2):
                continue
            # city_distance = np.linalg.norm(np.array(m)-np.array(m2))
            city_distance = distance.distance(m, m2).km  # /1000

            distances[city_name2] = city_distance
        new_cities[city_name] = {**city, **{"distances": distances}}
    return new_cities


# https://stackoverflow.com/questions/14432557/matplotlib-scatter-plot-with-different-text-at-each-data-point
def plot_cities(fig, ax, cities):
    x, y, n = [], [], []
    for city_name in cities.keys():
        city = cities[city_name]
        xy = city["xy"]
        x.append(xy[0])
        y.append(xy[1])
        n.append(city_name)
    ax.scatter(x, y)

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    return ax


def clear_plot(ax):
    ax.clear()


def plot_lines(fig, ax, cities):
    x, y, n = [], [], []

    for i in range(len(cities)):
        x.append(cities[i][0])
        y.append(cities[i][1])

    ax.plot(x, y, "-o")

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))
    return ax


def plot_animation(fig, ax, cities, lines, title=None, pause=0.1):
    clear_plot(ax)
    ax = plot_cities(fig, ax, cities)
    ax = plot_lines(fig, ax, lines)
    if (title is not None):
        ax.set_title(title)
    plt.draw()
    plt.pause(pause)


def to_list(cities):
    cities_list = []
    for city_name in cities.keys():
        city = cities[city_name]
        new_city = {**{"name": city_name}, **city}
        cities_list.append(new_city)
    return cities_list


def to_dict(cities_list):
    cities_dict = {}
    for i in range(len(cities_list)):
        cities_dict[cities_list[i]["name"]] = cities_list[i]
    return cities_dict


def generate_path(cities_list):
    xy = []
    for i in range(len(cities_list)):
        xy.append(cities_list[i]["xy"])
        # xy.append(cities_list[i]["xy"])
    # xy.append(cities_list[-1]["xy"])
    xy.append(cities_list[0]["xy"])
    return xy




def compute_cost(cities):
    cost = 0
    # Calculate the distance starting from the city with index 0 to the next city in the index
    for i in range(1, len(cities)):
        city0 = cities[i-1]
        city1 = cities[i]
        # print(city0["distances"])
        cost += city0["distances"][city1["name"]]
    # close the loop
    cost += cities[-1]["distances"][cities[0]["name"]]
    return cost

def generate_solution(cities, iter=3):
    new_cities = deepcopy(cities)
    for i in range(iter):
        # pick two cities in the path
        new_idx = random.sample(range(0, len(cities)), 2)
        # print(new_idx)
        # exchange their positions in the path
        tmp_city = deepcopy(new_cities[new_idx[0]])
        new_cities[new_idx[0]] = new_cities[new_idx[1]]
        new_cities[new_idx[1]] = tmp_city
    # return the new proposed path
    return new_cities

# Reference: https://gist.github.com/MNoorFawi/4dcf29d69e1708cd60405bd2f0f55700
def SA( cities, initial_temp=10000, cooling_rate=0.95,
        visualize=False, visualization_rate=0.01, fig=None, ax=None):
    cities_dict = cities
    cities = to_list(cities)
    time = 0
    temp = initial_temp
    costs = []
    temps = []
    new_solution_cost = 0
    current_solution = cities
    current_solution_cost = compute_cost(cities)
    new_solution = None
    while temp > 0.1:
        # Get new solution
        new_solution = generate_solution(current_solution)
        # Calculate the cost for the new solution
        new_solution_cost = compute_cost(new_solution)
        # Calculate p
        p = safe_exp((current_solution_cost - new_solution_cost)/temp)#(math.exp()
        # print(p)
        # if new solution is better or random less than p
        if(new_solution_cost < current_solution_cost or random.uniform(0,1) < p):
            current_solution = new_solution
            current_solution_cost = new_solution_cost
        if(visualize):
            path = generate_path(current_solution)
            title = f"Temp={temp:.3f}, Cost={current_solution_cost:.3f}\nInitial temp={initial_temp}, Cooling rate={cooling_rate}"
            plot_animation(fig, ax, cities_dict, path, pause=visualization_rate, title=title)

        # print(current_solution_cost)
        temp *= cooling_rate
        costs.append(current_solution_cost)
        temps.append(temp)
        time += 1
    return costs, temps, new_solution, new_solution_cost


def safe_exp(v):
    try:
        return math.exp(v)
    except:
        return 0






start = timeit.default_timer()
plt.figure(figsize=(15, 8))
parser = argparse.ArgumentParser()
parser.add_argument('--cooling', type=float, default=0.99)
parser.add_argument('--temp', type=int, default=10000)
args = parser.parse_args()



city_csv_file_path = "./city.csv"

visualize = True#False
visualization_rate = 0.001

initial_temp = args.temp #500
cooling_rate = args.cooling #0.999

csv_data = read_csv(city_csv_file_path)
cities = get_most_populated_cities(30, csv_data)
cities = get_coord_cities(cities)
cities = get_xy_gps(cities)

cities = get_distances(cities)


fig, ax = None, None
if(visualize):
    plt.ion()
    plt.show(block=True)
    fig, ax = plt.subplots()


start = timeit.default_timer()
costs, temps, new_solution, new_solution_cost = SA(cities, initial_temp, cooling_rate, visualize=visualize, visualization_rate=visualization_rate, fig=fig, ax=ax)
stop = timeit.default_timer()

print('Time: ', stop - start)
print(f"Final Cost: {costs[-1]}")
print(f"Number of steps: {len(costs)}")

path = generate_path(new_solution)
title = f"Final solution, Cost={costs[-1]:.3f}\nInitial temp={initial_temp}, Cooling rate={cooling_rate}"
plot_animation(fig, ax, cities, path, pause=5, title=title)
clear_plot(ax)
plt.plot([i for i in range(len(costs))], [c/costs[0]for c in costs], label="Cost")
plt.legend()
plt.plot([i for i in range(len(temps))], [t/temps[0] for t in temps], label="Temp")
plt.legend()
plt.title(f"Initial temp={initial_temp}, Cooling rate={cooling_rate}")
plt.draw()
plt.pause(10)
