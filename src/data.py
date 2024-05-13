from numpy import *
import matplotlib
import matplotlib.pyplot as plt

class data:
    cust = 8 #Number of customers
    m = 3 #Maximum number of vehicles
    Q = 10 #Maximum vehicle capacity

    # demand = array([4,  3,  3,   3,   3,  4,  3,  3  ]) #demand per customer
    demand = array([0, 4,  3,  3,   3,   3,  4,  3,  3 , 0]) #demand per customer
    twStart = array([0,  6,  8,  4,   6,   5,  6,  5,  4,  0]) #earlierst delivery time
    twEnd   = array([24, 6,  16, 20,  6,  19, 18, 19, 6, 24]) #latest delivery time


    # Travel cost matrix
    cost = array([
        [0,  7,  5,  3,  3,  4,  5,  4,  3,  0],
        [7,  0,  3,  5,  4, 11, 12, 10, 10,  7],
        [5,  3,  0,  5,  2,  9,  9,  8,  9,  5],
        [3,  5,  5,  0,  4,  6,  8,  7,  5,  3],
        [3,  4,  2,  4,  0,  6,  7,  6,  6,  3],
        [4, 11,  9,  6,  6,  0,  2,  2,  2,  4],
        [5, 12,  9,  8,  7,  2,  0,  1,  4,  5],
        [4, 10,  8,  7,  6,  2,  1,  0,  4,  4],
        [3, 10,  9,  5,  6,  2,  4,  4,  0,  3],
        [0,  7,  5,  3,  3,  4,  5,  4,  3,  0]
        ])

    #Travel time matrix
    timeCost = array([
        [0,  6,  6,  4,  4,  5,  6,  5,  4,  0],
        [6,  0,  4,  6,  5, 12, 13, 11, 11,  6],
        [6,  4,  0,  6,  3, 10, 10,  9, 10,  6],
        [4,  6,  6,  0,  5,  7,  9,  8,  6,  4],
        [4,  5,  3,  5,  0,  7,  8,  7,  7,  4],
        [5, 12, 10,  7,  7,  0,  3,  3,  3,  5],
        [6, 13, 10,  9,  8,  3,  0,  2,  5,  6],
        [5, 11,  9,  8,  7,  3,  2,  0,  5,  5],
        [4, 11, 10,  6,  7,  3,  5,  5,  0,  4],
        [0,  6,  6,  4,  4,  5,  6,  5,  4,  0]
        ])

    #The initial routes for Task 4
    #Each row describe the customers visited in a route. If the n'th index in a row is '1.0', then the route visits customer n.
    routes = array([
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 1, 1],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ])

    #The distance cost of the initial routes.
    costRoutes = array([15.0, 12.0, 22.0, 18.0, 15.0, 22.0, 18.0, 10.0, 15.0, 11.0, 13.0, 12.0])

    #For Task 5. Input the routes you found in Task 2
    routes = array([[0,0,0,0,1,1,0,1], [0,1,1,1,0,0,0,0], [1,0,0,0,0,0,1,0]])
    costRoutes = array([12.0, 13.0, 21.0])

    nodes = [(3.5,3.5),(0,0),(0,2),(3.5,1.5),(1,2.5),(5,5),(4,6.5),(3.5,5.5),(6,4)]
    labels = list(range(9))

    @staticmethod
    def plot_points(outputfile_name=None):
        "Plot instance points."
        style='bo'
        plt.plot([node[0] for node in data.nodes], [node[1] for node in data.nodes], style)
        plt.plot([data.nodes[0][0]], [data.nodes[0][1]], "rs")
        for (p, node) in enumerate(data.nodes):
            plt.text(node[0], node[1], '  '+str(data.labels[p]))
        plt.axis('scaled'); plt.axis('off')
        if outputfile_name is None:
            plt.show()
        else:
            plt.savefig(outputfile_name)

    @staticmethod
    def plot_routes(points, style='bo-'):
        "Plot lines to connect a series of points."
        for route in points:
            plt.plot(list(map(lambda p: data.nodes[p][0], route)), list(map(lambda p: data.nodes[p][1], route)), style)
        data.plot_points()


if __name__ == "__main__":
    data.plot_points()
    data.plot_routes([[0,1,2,3,0],[0,4,0],[0,7,6,5,8,0]])