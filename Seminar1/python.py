import matplotlib.pyplot as plt
import numpy as np

# plt.plot([1, 2, 3, 4])
# plt.ylabel('y-axis values')
# plt.show()

# 점 그래프
# plt.plot([1, 2, 3, 4], [1, 7, 5, 10], 'ro')
# plt.axis([0, 6, 0, 12])  # [x범위, y범위]
# plt.show()

# 막대 그래프
# plt.bar([1, 2, 3, 4], [1, 7, 5, 10])
# plt.axis([0, 6, 0, 12])
# plt.show()

# 여러 그래프(subplot())
# plt.figure(figsize=(10,10)) # 그래프 가로와 세로 크기
# plt.subplot(2,1,1)
# plt.title('dot graph')
# plt.plot([1,2,3,4,5], [1,7,5,10,11], 'ro')
# plt.axis([0,6,0,12])

# plt.subplot(2,1,2)
# plt.title('bar graph')
# plt.bar([1,2,3,4,5],[1,7,5,10,11])
# plt.axis([0,6,0,12])
# plt.show()

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle='--', label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()
