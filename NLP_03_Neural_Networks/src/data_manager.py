import numpy as np
from math import e
import altair as alt
import pandas as pd

class MainConcepts():
    def __init__(self):
        # self.visualize_sigmoid()
        # self.ex_one_layer()
        # self.visualize_tanh()
        # self.visualize_relu()
        self.xor_problem()

    def xor_problem(self):
        x1 = np.array([0,0,1,1])
        x2 = np.array([0,1,0,1])

        print('Can neural units compute simple functions of input?')
        print(f'x1 : {x1}')
        print(f'x2 : {x2}')
        print('-'*20)
        print(f'AND : {x1 & x2}')
        print(f'OR : {x1 | x2}')
        print(f'XOR : {(x1 != x2).astype(np.int32)}')

        # Perceptrons
        # A very simple neural unit
        # - binary output (0,1)
        # - no non-linear activation function

        # y = 0, if w*x + b <= 0
        # y = 1, if w*x + b < 0

        # [ expression if conditional else other thing for this many times ]

        print('-'*20)
        # What happens with input x
        def func_max(a):
            if a <= 0:
                return 0
            elif a > 0:
                return a
            else:
                return

    def ex_one_layer(self):
        # Suppose a unit has
        w = np.array([0.2, 0.3, 0.9])
        b = 0.5

        # What happens with input x
        x = np.array([0.5, 0.6, 0.1])

        # Compute sigmoid
        a = 1 / (1 + pow(e, -(np.sum(w*x) + b)))
        print(a)

    def visualize_relu(self):
        # Visualise ReLU - non-linear activation
        z = np.linspace(start=-10, stop=10, num=100)

        source = pd.DataFrame({
            'z': z,
            'f(z)': np.maximum(z, 0)
        })

        line = alt.Chart(source).mark_line().encode(
            x='z',
            y='f(z)'
        ).properties(
            width=800,
            height=300,
            title='Non-linear activation : ReLU'
        )

        text = line.mark_text(
            text="y = max(z,0)",
            dx=-20, dy=-5,
            fontSize=20
        ).encode(
            x=alt.datum(-4), y=alt.datum(0.8)
        )

        chart = line + text

        chart.save('./fig/relu_visualize.png')


    def visualize_tanh(self):
        # Visualise tanh function - non linear activation
        z = np.linspace(start=-10, stop=10, num=100)

        source = pd.DataFrame({
            'z': z,
            'f(z)': (pow(e, z) - pow(e, -z)) / (pow(e, z) + pow(e, -z))
        })

        line = alt.Chart(source).mark_line().encode(
            x='z',
            y='f(z)'
        ).properties(
            width=800,
            height=300,
            title='Non-linear activation : tanh'
        )

        text = line.mark_text(
            text="y = (e^z - e^-z) / (e^z + e^-z)",
            dx=-20, dy=-5,
            fontSize=20
        ).encode(
            x=alt.datum(-4), y=alt.datum(0.8)
        )

        chart = line + text

        chart.save('./fig/tanh_visualize.png')

    def visualize_sigmoid(self):
        # Visualise sigmoid function - non linear activation
        z = np.linspace(start=-8, stop=8, num=100)

        source = pd.DataFrame({
            'z': z,
            'f(z)': 1 / (1 + pow(e, -z))
        })

        line = alt.Chart(source).mark_line().encode(
            x='z',
            y='f(z)'
        ).properties(
            width=800,
            height=300,
            title='Sigmoid'
        )

        text = line.mark_text(
            text="y = 1 / (1 + e^-z)",
            dx=-20, dy=-5,
            fontSize=20
        ).encode(
            x=alt.datum(-4), y=alt.datum(0.8)
        )

        chart = line + text

        chart.save('./fig/sigmoid_visualize.png')