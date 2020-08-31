import random
import math
from random import randrange
from decimal import *
import statistics
import time


class SA(object):
    """docstring for SA"""

    def __init__(self):
        super(SA, self).__init__()

    def setData(self, values, weights, W, T):
        self.values = values
        self.weights = weights
        self.W = W
        self.T = T
        self.length = len(values) - 1
        self.solution = []
        for x in self.values:
            self.solution.append(0)

    def detail(self, solution):
        value = 0
        weight = 0
        for index, bit in enumerate(solution):
            if bit:
                value += self.values[index]
                weight += self.weights[index]
        return value, weight

    def popIn(self, solution):
        i = 0
        while i < self.length:
            if solution[i] == 0:
                break
            else:
                i += 1
        if i == self.length:
            return 0

        popIn = random.randint(0, self.length - 1)
        while solution[popIn]:
            popIn = random.randint(0, self.length - 1)
        return popIn

    def popOut(self, solution):
        lis = []
        for index, bit in enumerate(solution):
            if bit:
                lis.append(index)
        return random.choice(lis)

    def fix(self, solution):
        detail = self.detail(solution)

        while detail[1] > self.W:
            solution[self.popOut(solution)] = 0
            detail = self.detail(solution)

        popIn = self.popIn(solution)
        while detail[1] + self.weights[popIn] <= self.W:
            solution[popIn] = 1
            popIn = self.popIn(solution)
            detail = self.detail(solution)

        return solution

    def generator(self):
        getcontext().prec = 6
        self.newSolution = self.solution.copy()
        self.newSolution[self.popIn(self.newSolution)] = 1
        self.newSolution = self.fix(self.newSolution)
        detail = self.detail(self.newSolution)

        oldDetail = self.detail(self.solution)
        delta = detail[0] - oldDetail[0]
        p = pow(Decimal(math.e), (Decimal(delta) / Decimal(self.T)))
        if delta > 0 or random.uniform(0, 1) < p:
            self.solution = self.newSolution.copy()

    def run(self):
        while self.T > 0:
            self.generator()
            self.T -= 1
        detail = self.detail(self.solution)
        print(detail[0])


value = [135, 139, 149, 150, 156, 163, 173, 184, 192, 201, 210, 214, 221, 229, 240]
weight = [70, 73, 77, 80, 82, 87, 90, 94, 98, 106, 110, 113, 115, 118, 120]
W = 750
T = 100
SA = SA()
SA.setData(value, weight, W, T)
SA.run()