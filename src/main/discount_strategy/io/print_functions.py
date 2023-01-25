import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
path_to_util = os.path.join((Path(os.path.abspath(__file__)).parents[2]), "util")
sys.path.insert(1, path_to_util)
import constants
path_to_images = constants.PATH_TO_IMAGES

class Painter:

    # print red vertexes for customers  without discount, green - with discount, blue - PUP
    def printVertex(self, instance):

        plt.figure(0)
        ax, fig = plt.gca(), plt.gcf()
        ax.set_aspect(1.0)

        #P_choose_disc_option = 100 - float(instance.name.split('_')[len(instance.name.split('_'))-1])

        #ax.set_title("Instance: " + str(instance.name) + "\n" + \
        #    "Number of customers: " + str(instance.NR_CUST) + ". P_choose_disc_option: "+str(P_choose_disc_option) )

        for customer in instance.customers:
            ax.scatter(customer.xCoord, customer.yCoord, marker='o', s=40, color='red')
            ax.text(customer.xCoord + 0.35,customer.yCoord + 0.35, customer, fontsize=12)

        ax.scatter(instance.depot.xCoord, instance.depot.yCoord, marker='^', s=60, color='blue')
        ax.text(instance.depot.xCoord + 0.35,instance.depot.yCoord + 0.35, instance.depot, fontsize=12)

        ax.scatter(instance.pup.xCoord, instance.pup.yCoord, marker='s', s=60, color="blue")
        ax.text(instance.pup.xCoord + 0.35, instance.pup.yCoord + 0.35, instance.pup, fontsize=12)
        plt.show()

    def printVertexDisc(self, instance, policy):

        plt.figure(0)
        ax, fig = plt.gca(), plt.gcf()
        ax.set_aspect(1.0)

        # str(round(instance.FLAT_RATE_SHIPPING_FEE,2))"\n" + \
        #ax.set_title("Instance: " + str(instance.name)+ "\n" + \
        #    "Number of customers: " + str(instance.NR_CUST) +\
        #             "    Discount cost: " + r"$0.3\cdot d^{max}$" )


        for customer in instance.customers:
            if (1<<(customer.id-1))&policy:

                wd = ax.scatter(customer.xCoord, customer.yCoord, marker='o', s=45, color='red', label='customer with incentive')
                #ax.text(customer.xCoord + 0.4, customer.yCoord + 0.2, customer, fontsize=10)
            else:
                nd = ax.scatter(customer.xCoord, customer.yCoord, marker='o', s=35, color='white', edgecolors='black', label = "customer")
                #ax.text(customer.xCoord + 0.4, customer.yCoord + 0.2, customer, fontsize=10)



        d = ax.scatter(instance.depot.xCoord, instance.depot.yCoord, marker='s', s=80, color='blue')
        #ax.text(instance.depot.xCoord + 0.35,instance.depot.yCoord + 0.35, instance.depot, fontsize=12)

        p = ax.scatter(instance.pup.xCoord, instance.pup.yCoord, marker='^', s=80, color="blue")
        #ax.text(instance.pup.xCoord - 1.35, instance.pup.yCoord - 0.55, instance.pup, fontsize=12)


        if policy ==0:
            plt.legend((nd, d, p),
                           ('No Discount', 'Depot', 'PUP'),
                           scatterpoints=1,
                           loc='lower left',
                           ncol=2,
                           fontsize=8)
        elif policy == 2**instance.NR_CUST -1:
            plt.legend((wd, d, p),
                       ('With Discount',  'Depot', 'PUP'),
                       scatterpoints=1,
                       loc='lower left',
                       ncol=2,
                       fontsize=8)
        else:
            plt.legend((wd, nd, d, p),
                       ('Offered incentive', 'No incentive', 'Depot', 'Pickup point'),
                       scatterpoints=1,
                       bbox_to_anchor=(-0.01,1), loc="lower left",
                       ncol=2,
                       fontsize=11)

        plt.savefig(os.path.join(path_to_images, 'map.eps'), transparent=False,
                    bbox_inches='tight')

        plt.show()

    def printConvergence(self, instance, time, lbPrint, ubPrint, bab_obj):
        fig = plt.figure()
        #plt.xscale('log')
        plt.style.use('seaborn-whitegrid')
        plt.plot(time, lbPrint, '--', label=r'lower bound (not valid for policy)')
        plt.plot(time, ubPrint, '-', label=r'best policy upper bound ($S_*^U$)')
        gap = []
        for i, lb in enumerate(lbPrint):
            gap.append((ubPrint[i]-lbPrint[i])/ubPrint[i])
        #plt.plot(time, gap, '-', label='Gap')
        #plt.plot(time, lbPrint, '-',alpha=0.5, label=r'lbCurrent')
        #plt.plot(time, ubPrint, '-', alpha=0.5, label=r'ubCurrent')

        plt.axhline(y = bab_obj, color='r', linestyle='--', label = 'solution objective')
        plt.title("A convergency of B&B for " + str(instance.NR_CUST) + " customers. Instance: " + instance.name[:-4] )
        plt.xlabel("time (s)")
        plt.ylabel("policy cost")


        plt.show()






def print_route(x, fig, ax, sets, n, nodes):
    use = []
    for v in sets['V']:
        for i in sets['0N2']:
            for j in sets['0N2']:
                if x[i, j, v] == 1:
                    if v not in use:
                        use.append(v)
    cycol = cycle('bgrcmk')
    for v in use:
        start = 0
        finish = 0
        sequence = [0]
        while start != n + 2:
            while (x[start, finish, v]) < 0.5:
                finish += 1
            sequence.append(finish)
            start = finish
            finish = 1
        color = next(cycol)
        draw(sequence, color, v, fig, ax, nodes)


def print_route2(x, fig, ax, n, nodes):
    use = []

    start = 0
    finish = 0
    sequence = [0]
    for i in range(1, n + 1 + fleet_size):
        if x[0, i] == 1 and i not in sequence:
            start = i
            sequence.append(start)
            while start != 0:
                while (x[start, finish]) < 0.5:
                    finish += 1
                    if finish == n + 1 + fleet_size:
                        finish = 0
                sequence.append(finish)
                start = finish
                finish = 1
    draw(sequence, "blue", 1, fig, ax, nodes)


def draw(sequence, color, v, fig, ax, nodes):
    # plt.axis('equal')
    x = np.arange(len(sequence))
    y = np.arange(len(sequence))
    for i in range(len(sequence) - 1):
        x[i] = nodes[sequence[i]]['x']
        y[i] = nodes[sequence[i]]['y']
    x[len(sequence) - 1] = x[0]
    y[len(sequence) - 1] = y[0]
    for i in range(0, len(x), 1):
        ax.plot(x[i:i + 2], y[i:i + 2], c=color)
    ax.plot(x[0:2], y[0:2], c=color, label=v)

    # leg = plt.legend()


def set_legend(fig):
    customers_without = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                      markersize=10, label='Customers without discount')
    customers_with = mlines.Line2D([], [], color='green', marker='s', linestyle='None',
                                   markersize=10, label='Customers with discount')
    pup = mlines.Line2D([], [], color='blue', marker='s', linestyle='None',
                        markersize=10, label='PUP')
    # ax.legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=16, handles=[customers_without, customers_with, pup])
    fig.legend(handles=[customers_without, customers_with, pup], loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=3)
    # fig.legend(handles=[customers_without, customers_with, pup])
    return fig