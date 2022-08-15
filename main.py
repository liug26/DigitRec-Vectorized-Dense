import math
import csvreader
import neuralnetwork
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
x, y, y2 = [], [], []


def main():
    global x, y, y2
    # x (#inputs, m), y (1, m), y2 (#outputs, m)
    x, y, y2 = csvreader.read()
    learn(start=0, end=1024*30, batch_size=1024, num_iterations=100)
    test(start=0, end=1024*30)
    test(start=1024*30, end=36360)
    test(start=36360, end=42000)

    x_points = np.array(range(len(neuralnetwork.cost)))
    y_points = np.array(neuralnetwork.cost)
    plt.plot(x_points, y_points)
    plt.show()


def learn(start, end, batch_size, num_iterations):
    num_batches = math.ceil((end-start) / batch_size)
    progress_bar = tqdm(total=num_batches * num_iterations)
    for b in range(num_batches):
        for t in range(num_iterations):
            neuralnetwork.learn(x[:, start + batch_size * b : min(end, start + batch_size * (b+1))],
                                y2[:, start + batch_size * b : min(end, start + batch_size * (b+1))], t)
            progress_bar.update(1)
    progress_bar.close()


def test(start, end):
    neuralnetwork.test(x[:, start:end], y[:, start:end])


if __name__ == '__main__':
    main()
