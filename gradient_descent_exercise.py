import numpy as np
import pandas as pd
import math



def gradient_descent_exercise(x,y):
    m_curr = b_curr = 0
    n = len(x)
    learning_rate = 0.0001
    prev_cost = None
    i = 0

    while True:
        i += 1
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val ** 2 for val in (y-y_predicted)])
        md = -(2/n) * sum(x*(y-y_predicted))
        bd = -(2/n) * sum((y-y_predicted))
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iterations: {}".format(m_curr, b_curr, cost, i))

        if prev_cost is not None and math.isclose(prev_cost, cost, rel_tol=1e-20):
            print("Converged at iteration {}".format(i))
            print("m {}, b {}, cost {}".format(m_curr, b_curr, cost))
            break
        prev_cost = cost

df = pd.read_csv("data/test_score.csv")

x = df['math']
y = df['cs']

gradient_descent_exercise(x,y)
