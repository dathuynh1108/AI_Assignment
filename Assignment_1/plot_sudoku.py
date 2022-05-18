from cProfile import label
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

testcases = ["1", "2", "3", "4", "5"]
time_bfs = [0.27359890937805176, 0.4240550994873047, 0.1387341022491455, 0.22034025192260742, 10.842854976654053]
time_heuristic = [0.01746392250061035, 0.01584482192993164, 0.027732133865356445,  0.05154228210449219, 0.09053993225097656]

mem_bfs = [26436, 31160, 25356, 40008, 220272]
mem_heutistic = [23508, 23560, 23628, 23736, 37352]


# fig, ax = plt.subplots()
# ax.plot(testcases, time_bfs, 'b')
# ax.plot(testcases, time_heuristic, 'r')
# heuristic = mpatches.Patch(color='red', label='Heuristic search')
# bfs = mpatches.Patch(color='blue', label='Breadth-first search')
# ax.set_xlabel('Level')
# ax.set_ylabel('Thời gian (giây)')
# ax.legend(handles=[heuristic, bfs])

fig, ax = plt.subplots()
ax.plot(testcases, mem_bfs, 'b')
ax.plot(testcases, mem_heutistic, 'r')
heuristic = mpatches.Patch(color='red', label='Heuristic search')
bfs = mpatches.Patch(color='blue', label='Breadth-first search')
ax.set_xlabel('Level')
ax.set_ylabel('Bộ nhớ (KB)')
ax.legend(handles=[heuristic, bfs])


plt.show()