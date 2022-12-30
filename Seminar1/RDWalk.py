import numpy as np
import matplotlib.pyplot as plt

probability = [0.5, 0.5]    # [up, down]

# starting point
start = 0
rand_walks = [start]

# creating the random points
rand_point = np.random.random(400)
down_probability = rand_point < probability[0]
up_probability = rand_point >= probability[1]

# random walk process
# z(t) = z(t-1) + a(t), where a(t) is white noise
for down, up in zip(down_probability, up_probability):
    rand_walks.append(rand_walks[-1] - down + up)

print(rand_walks)
plt.plot(rand_walks)
plt.show()
