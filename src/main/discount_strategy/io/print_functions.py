import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from src.main.discount_strategy.util import constants
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
        for pup in instance.pups:
            ax.scatter(pup.xCoord, pup.yCoord, marker='s', s=60, color="blue")
            ax.text(pup.xCoord + 0.35, pup.yCoord + 0.35, pup, fontsize=12)
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
            #dict_name_clustered = {15:1, 11:2, 13:3, 7:4, 8:5, 9:6, 10:7}
            dict_name_random = {11: 1, 9: 2, 1: 3, 14: 4, 15: 5, 4: 6}
            if customer.id < 20:

                if (1<<(customer.id-1))&policy:

                    wd = ax.scatter(customer.xCoord, customer.yCoord, marker='o', s=60, color='black', label='customer with incentive')

                    ax.text(customer.xCoord + 1, customer.yCoord +0.6, dict_name_random[customer.id], fontsize=12)
                else:
                    nd = ax.scatter(customer.xCoord, customer.yCoord, marker='o', s=60, color='white', edgecolors='black', label = "customer")
                    #ax.text(customer.xCoord + 0.24, customer.yCoord + 0.5, customer, fontsize=8)
                    #if customer.id in [7,8,10]:
                    #   ax.text(customer.xCoord  + 1, customer.yCoord +0.6, dict_name[customer.id], fontsize=12)
                    #if customer.id in [9]:
                    #    ax.text(customer.xCoord -0.7, customer.yCoord +0.8, dict_name[customer.id], fontsize=12)
                    #else:
                    #   ax.text(customer.xCoord + 0.24, customer.yCoord + 0.5, customer, fontsize=14)



        d = ax.scatter(instance.depot.xCoord, instance.depot.yCoord, marker='s', s=90, color='blue')
        #ax.text(instance.depot.xCoord -3,instance.depot.yCoord -5, "depot", fontsize=14)

        for pup in instance.pups:
            p = ax.scatter(pup.xCoord, pup.yCoord, marker='^', s=100, color="blue")
            #ax.text(pup.xCoord +6, pup.yCoord - 1, "pickup point", fontsize=14)
        ax.set_xlim(None, 59)
        ax.set_ylim(None, 69)
        #ax.text(instance.pup.xCoord - 1.35, instance.pup.yCoord - 0.55, instance.pup, fontsize=12)
        # ax.legend((nd,wd,  d, p),     ('No discount','With discount', 'Depot', 'Pickup point'),
        #                               scatterpoints=1,
        #                               ncol=1,
        #                               fontsize=12, bbox_to_anchor=(1,0), loc="lower right")
        # # if policy ==0:
        #     plt.legend((nd, d, p),
        #                    ('No Discount', 'Depot', 'PUP'),
        #                    scatterpoints=1,
        #                    loc='lower left',
        #                    ncol=2,
        #                    fontsize=8)
        # elif policy == 2**instance.NR_CUST -1:
        #     plt.legend((wd, d, p),
        #                ('With Discount',  'Depot', 'PUP'),
        #                scatterpoints=1,
        #                loc='lower left',
        #                ncol=2,
        #                fontsize=8)
        # else:
        #     plt.legend((wd, nd, d, p),
        #                ('Offered incentive', 'No incentive', 'Depot', 'Pickup point'),
        #                scatterpoints=1,
        #                bbox_to_anchor=(-0.01,1), loc="lower left",
        #                ncol=2,
        #                fontsize=11)

        plt.savefig(os.path.join(path_to_images, 'map_example_random.eps'), transparent=False,
                   bbox_inches='tight')

        plt.show()
    def printVertexDiscTemp(self, instance_clustered, policy_clustered, instance_random, policy_random):

        fig, axes = plt.subplots(1, 2, sharey=True, width_ratios=[1, 2],  figsize=(10, 5))
        #axes[0].set_aspect(1.0)
        #axes[1].set_aspect(1.0)
        # str(round(instance.FLAT_RATE_SHIPPING_FEE,2))"\n" + \
        #ax.set_title("Instance: " + str(instance.name)+ "\n" + \
        #    "Number of customers: " + str(instance.NR_CUST) +\
        #             "    Discount cost: " + r"$0.3\cdot d^{max}$" )


        for customer in instance_random.customers:
            #dict_name_clustered = {15:1, 11:2, 13:3, 7:4, 8:5, 9:6, 10:7}
            dict_name_random = {11: 1, 9: 2, 1: 3, 14: 4, 15: 5, 4: 6}
            if customer.id < 20:

                if (1<<(customer.id-1))&policy_random:

                    wd = axes[0].scatter(customer.xCoord, customer.yCoord, marker='o', s=60, color='black', label='customer with incentive')

                    axes[0].text(customer.xCoord + 1, customer.yCoord +0.6, dict_name_random[customer.id], fontsize=14)
                else:
                    nd = axes[0].scatter(customer.xCoord, customer.yCoord, marker='o', s=60, color='white', edgecolors='black', label = "customer")
                    #ax.text(customer.xCoord + 0.24, customer.yCoord + 0.5, customer, fontsize=8)
                    #if customer.id in [7,8,10]:
                    #   ax.text(customer.xCoord  + 1, customer.yCoord +0.6, dict_name[customer.id], fontsize=12)
                    #if customer.id in [9]:
                    #    ax.text(customer.xCoord -0.7, customer.yCoord +0.8, dict_name[customer.id], fontsize=12)
                    #else:
                    #   ax.text(customer.xCoord + 0.24, customer.yCoord + 0.5, customer, fontsize=14)
        axes[0].scatter(instance_random.depot.xCoord, instance_random.depot.yCoord, marker='s', s=90,
                          color='blue')
        # ax.text(instance.depot.xCoord -3,instance.depot.yCoord -5, "depot", fontsize=14)

        for pup in instance_random.pups:
            p = axes[0].scatter(pup.xCoord, pup.yCoord, marker='^', s=100, color="blue")
        for customer in instance_clustered.customers:
            dict_name_clustered = {15:1, 11:2, 13:3, 7:4, 8:5, 9:6, 10:7}
            if customer.id < 20:

                if (1<<(customer.id-1))&policy_clustered:

                    wd = axes[1].scatter(customer.xCoord, customer.yCoord, marker='o', s=60, color='black', label='customer with incentive')

                    axes[1].text(customer.xCoord + 1, customer.yCoord +0.6, dict_name_clustered[customer.id], fontsize=14)
                else:
                    nd = axes[1].scatter(customer.xCoord, customer.yCoord, marker='o', s=60, color='white', edgecolors='black', label = "customer")
                    #ax.text(customer.xCoord + 0.24, customer.yCoord + 0.5, customer, fontsize=8)
                    if customer.id in [7,8,10]:
                       axes[1].text(customer.xCoord  + 1, customer.yCoord +0.6, dict_name_clustered[customer.id], fontsize=14)
                    if customer.id in [9]:
                        axes[1].text(customer.xCoord -0.7, customer.yCoord +0.8, dict_name_clustered[customer.id], fontsize=14)
                    #else:
                    #   ax.text(customer.xCoord + 0.24, customer.yCoord + 0.5, customer, fontsize=14)


        d = axes[1].scatter(instance_clustered.depot.xCoord, instance_clustered.depot.yCoord, marker='s', s=90, color='blue')
        #ax.text(instance.depot.xCoord -3,instance.depot.yCoord -5, "depot", fontsize=14)

        for pup in instance_clustered.pups:
            p = axes[1].scatter(pup.xCoord, pup.yCoord, marker='^', s=100, color="blue")
            #ax.text(pup.xCoord +6, pup.yCoord - 1, "pickup point", fontsize=14)
        axes[0].set_xlim(None, 59)
        axes[0].set_ylim(None, 69)
        #ax.text(instance.pup.xCoord - 1.35, instance.pup.yCoord - 0.55, instance.pup, fontsize=12)
        axes[1].legend((nd,wd,  d, p),     ('No discount','With discount', 'Depot', 'Pickup point'),
                                      scatterpoints=1,
                                      ncol=1,
                                      fontsize=14, bbox_to_anchor=(1,0), loc="lower right")
        # # if policy ==0:
        #     plt.legend((nd, d, p),
        #                    ('No Discount', 'Depot', 'PUP'),
        #                    scatterpoints=1,
        #                    loc='lower left',
        #                    ncol=2,
        #                    fontsize=8)
        # elif policy == 2**instance.NR_CUST -1:
        #     plt.legend((wd, d, p),
        #                ('With Discount',  'Depot', 'PUP'),
        #                scatterpoints=1,
        #                loc='lower left',
        #                ncol=2,
        #                fontsize=8)
        # else:
        #     plt.legend((wd, nd, d, p),
        #                ('Offered incentive', 'No incentive', 'Depot', 'Pickup point'),
        #                scatterpoints=1,
        #                bbox_to_anchor=(-0.01,1), loc="lower left",
        #                ncol=2,
        #                fontsize=11)

        plt.savefig(os.path.join(path_to_images, 'map_example_random_clustered.eps'), transparent=False,
                   bbox_inches='tight')

        plt.show()

    def printConvergence(self, instance, time, lbPrint, ubPrint, bab_obj):
        fig = plt.figure()
        #plt.xscale('log')
        lbPrint_new = [lbPrint[-1]]
        for i in lbPrint[len(lbPrint)-2::-1]:
            lbPrint_new.append(min(i, lbPrint_new[-1]))
        plt.style.use('seaborn-whitegrid')
        plt.plot(time, lbPrint_new[::-1], '--', label=r'lower bound (not valid for policy)')
        plt.plot(time, ubPrint, '-', label=r'best policy upper bound ($S_*^U$)')
        gap = []
        for i, lb in enumerate(lbPrint):
            gap.append((ubPrint[i]-lbPrint[i])/ubPrint[i])
        #plt.plot(time, gap, '-', label='Gap')
        #plt.plot(time, lbPrint, '-',alpha=0.5, label=r'lbCurrent')
        #plt.plot(time, ubPrint, '-', alpha=0.5, label=r'ubCurrent')

        plt.axhline(y = bab_obj, color='r', linestyle='--', label = 'solution objective', alpha = 0.3)
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