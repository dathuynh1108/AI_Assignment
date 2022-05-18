from cProfile import label
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

testcases = ["1", "2", "3", "4", "5"]
time_bfs = [0.0104827880859375, 0.07595634460449219, 2.3230483531951904, 2.7679717540740967, 18.47396945953369]
time_heuristic = [0.008929252624511719, 0.06661629676818848, 1.2883100509643555, 1.5241098403930664, 1.678342580795288]

mem_bfs = [23956, 23604,36788, 38060, 100496]
mem_heutistic = [23477, 24316, 32620, 32929, 31032]


# fig, ax = plt.subplots()
# ax.plot(testcases, time_bfs, 'b')
# ax.plot(testcases, time_heuristic, 'r')
# heuristic = mpatches.Patch(color='red', label='Heuristic search')
# bfs = mpatches.Patch(color='blue', label='Breadth-first search')
# # ax.set_title("So sánh thời gian giữa 2 giải thuật")
# ax.set_xlabel('Level')
# ax.set_ylabel('Thời gian (giây)')
# ax.legend(handles=[heuristic, bfs])

fig, ax = plt.subplots()
ax.plot(testcases, mem_bfs, 'b')
ax.plot(testcases, mem_heutistic, 'r')
heuristic = mpatches.Patch(color='red', label='Heuristic search')
bfs = mpatches.Patch(color='blue', label='Breadth-first search')
# ax.set_title("So sánh thời gian giữa 2 giải thuật")
ax.set_xlabel('Level')
ax.set_ylabel('Bộ nhớ (KB)')
ax.legend(handles=[heuristic, bfs])


plt.show()