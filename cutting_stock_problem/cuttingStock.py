import copy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches




class Rectangle:
    def __init__(self, height, width, value):
        self.height = min(width, height)
        self.width = max(width, height)
        self.value = value
        self.position = None

    def __repr__(self):
        return f'Rectangle height: {self.height}, width: {self.width}, ' + \
               f'value: {self.value}'


class Data:
    def __init__(self, radius, rectangles):
        self.radius = radius
        self.rectangles = rectangles
        self.n_rectangles = len(rectangles)

    def fitness_function(self, x):
        all_rectangles = [copy.deepcopy(self.rectangles[i]) for i, n in
                      enumerate(x) for j in range(n)]
        place_in_circle(all_rectangles, self.radius)
        value = 0
        for r in all_rectangles:
            value += r.value
            if r.position is None:
                return 0
        return value


class DataLoader:
    def __init__(self, dir='cutting'):
        self.dir = dir

    def load(self, filename):
        df = pd.read_csv(os.path.join(self.dir, filename), header=None)
        radius = int(filename.replace('r', '').replace('.csv', ''))
        rectangles = []
        for i, row in df.iterrows():
            rectangles.append(Rectangle(row[0], row[1], row[2]))
        return Data(radius=radius, rectangles=rectangles)


def pitagoras(a, b, c):
    if a is None:
        return np.sqrt(c ** 2 - b ** 2)
    if b is None:
        return np.sqrt(c ** 2 - a ** 2)
    if c is None:
        return np.sqrt(a ** 2 + b ** 2)


def get_row_width(y, radius):
    return 2 * pitagoras(y, None, radius)


def draw_rectangles(rectangles, radius, filename):
    figure, axes = plt.subplots()
    circle = plt.Circle((0, 0), radius, fill=False)
    axes.set_aspect(1)
    axes.add_artist(circle)
    plt.xlim(-radius, radius)
    plt.ylim(-radius, radius)
    for rect in rectangles:
        if rect.position is None:
            continue
        r = patches.Rectangle(xy=rect.position, width=rect.width,
                              height=rect.height, linewidth=1, edgecolor='g',
                              facecolor='none')
        axes.add_patch(r)
    plt.savefig(f'plots\\{filename}_positions.png')


def place_in_circle(rectangles, radius):
    y = pitagoras(rectangles[0].width / 2, None, radius)

    rectangles[0].position = (
    -rectangles[0].width / 2, y - rectangles[0].height)

    current_row = [rectangles[0]]
    prev_rect = rectangles[0]
    for i in range(1, len(rectangles)):

        for j in range(i - 1, -1, -1):
            if rectangles[j].position is not None:
                prev_rect = rectangles[j]
                break

        if fit_horizontal(prev_rect, radius, rectangles[i]):
            position = calculate_standard_position(prev_rect,
                                                   rectangles[i])
        else:
            max_height = max(rect.height for rect in current_row)
            current_row = []
            position = calculate_position_in_new_row(
                max_height, prev_rect, rectangles[i], radius)

        if fit_by_left_side(position, rectangles[i], radius):
            rectangles[i].position = position
        else:
            x = max(-pitagoras(position[1], None, radius),
                    -pitagoras(position[1] + rectangles[i].height, None, radius))
            if check_if_in_circle(rectangles[i], (x, position[1]), radius):
                rectangles[i].position = (x, position[1])

        current_row.append(rectangles[i])


def fit_horizontal(previous_rect, radius, rect):
    prev_upper_right_corner = (previous_rect.position[0] + previous_rect.width,
                               previous_rect.position[
                                   1] + previous_rect.height)
    prev_lower_right_corner = (previous_rect.position[0] + previous_rect.width,
                               previous_rect.position[1])

    max_upper_x = pitagoras(prev_upper_right_corner[1], None, c=radius)
    max_lower_x = pitagoras(prev_upper_right_corner[1] - rect.height,
                            None, c=radius)

    rect_right_x = prev_lower_right_corner[0] + rect.width
    if rect_right_x < max_lower_x and rect_right_x < max_upper_x:
        return True
    else:
        return False


def fit_by_left_side(rect_position, rect, radius):
    if is_in_circle(rect_position[0] + rect.width, rect_position[1],
                    radius) and is_in_circle(rect_position[0] + rect.width,
                                             rect_position[1] + rect.height,
                                             radius):
        return True
    return False


def calculate_standard_position(previous_rect, rect):
    prev_upper_right_corner = (previous_rect.position[0] + previous_rect.width,
                               previous_rect.position[
                                   1] + previous_rect.height)
    return (prev_upper_right_corner[0],
            prev_upper_right_corner[1] - rect.height)


def calculate_position_in_new_row(prev_row_max_height, prev_rect, rect,
                                  radius):
    rect_upper_y = prev_rect.position[
                       1] + prev_rect.height - prev_row_max_height
    rect_left_x = -pitagoras(rect_upper_y, None, radius)

    rect_lower_y = rect_upper_y - rect.height
    if is_in_circle(rect_left_x, rect_upper_y, radius) and is_in_circle(
            rect_left_x, rect_lower_y, radius):
        return rect_left_x, rect_lower_y
    else:
        return max(-pitagoras(rect_upper_y, None, radius),
                   -pitagoras(rect_lower_y, None, radius)), rect_lower_y


def check_if_in_circle(rect, position, radius):
    if is_in_circle(position[0], position[1], radius) and is_in_circle(position[0], position[1] + rect.height, radius) and is_in_circle(position[0] + rect.width, position[1], radius) and is_in_circle(position[0] + rect.width, position[1] + rect.height, radius):
        return True
    return False


def is_in_circle(x, y, radius):
    return x ** 2 + y ** 2 <= radius ** 2




