'''


def disc_cost_vs_tsp_savings(instances_num, instances ):

    n = instances["n"]
    file_name = instances["file_name"]
    pup_node = instances["pup_node"]
    shipping_price = instances["shipping_price"]
    discount_2_segment = instances["discount_2_segment"]


    gamma = 0.1
    discount = shipping_price

    file_results = time.strftime("%Y-%m-%d") + " output_disc_vs_tsp " + str(n) + " " + file_name
    head = ['Alpha(reduction)', 'Policy', 'Exp_cost', '#_discounts']


    sets, input_data = initialise_VRP(n, file_name, pup_node)
    lowest_discount = {}
    lowest_discount[0] = 0 * shipping_price
    lowest_discount[1] = discount_2_segment * shipping_price
    lowest_discount[2] = constants.BIGM * shipping_price

    input_data['lowest_discount'] = lowest_discount
    input_data['discount_2_segment'] = discount_2_segment
    input_data['segment_probability'] =  set_segment_probability(10,80,10)

    # set choice scenarios "1" denote PUP delivery, "0" denotes home delivery
    W = set_scenarios(n, 0, 1)
    C = {}
    for w in W:
        C[w] = classic_VRP_for_scenario_without_TW(W[w], sets, input_data)
    c_baseline = VRP_visit_all_nodes(W[0], sets, input_data)[0]


    tsp_cost = []
    for w in C:
        tsp_cost.append(C[w])
    tsp_cost.sort()
    vertical_line_w = 0
    for w in tsp_cost:
        if w < c_baseline*0.8:
            vertical_line_w += 1
            horizontal_line = w
    plt.figure(0)
    #plt.plot(tsp_cost, label=str(n) + ' customers ' + file_name)
    #plt.ylabel('tsp cost')
    #plt.xlabel('scenario, #')
    #plt.show()

    plt.hist(tsp_cost, bins=20)
    plt.ylabel('# of scenarios')
    plt.xlabel('TSP cost')

    plt.savefig(os.path.join(file_images, time.strftime("%Y-%m-%d-%H-%M")+file_name+str(n)+' TSP_costs.png'))
    plt.show()

    '''
'''
    tsp_cost_x = []
    num_discounts_y = []
    discount_size_y = []
    tsp_and_discount_cost_x = []
    #set discount policies
    Y = set_scenarios(n, 0, 1)
    prob_matrix = set_probability_matrix(Y, W, input_data, sets['N'])

    dict_allowed_y ={}
    min_disc_size = {}
    for reduction in [ 0, 0.025, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]:
        dict_allowed_y[reduction] = []
        min_disc_size[reduction] = n
        for y in Y:
            cumm_prob_expensive = 0
            for w in C:
                #if C[w] + disc_cost(Y[y], W[w], discount) > c_baseline * (1 - reduction):
                if C[w] > c_baseline*(1-reduction):
                    cumm_prob_expensive += prob_matrix[y][w]
            if cumm_prob_expensive < gamma:
                disc = 0
                dict_allowed_y[reduction].append(Y[y])

                for i in sets["N"]:
                    if Y[y][i] == 1:
                        disc +=1
                if disc < min_disc_size[reduction]:
                    min_disc_size[reduction] = disc
        if min_disc_size[reduction]< n:
            tsp_cost_x.append((1-reduction)*c_baseline)
            tsp_and_discount_cost_x.append((1-reduction)*c_baseline +  min_disc_size[reduction]*discount )
            num_discounts_y.append(min_disc_size[reduction])

    plt.figure(1)
    plt.scatter(tsp_cost_x, num_discounts_y, c = "red")
    plt.ylabel('min # of discounts')
    plt.xlabel('TSP cost (worst case)')
    plt.savefig(os.path.join(file_images, time.strftime("%Y-%m-%d-%H-%M") + 'TSP vs discount'+str(n) +file_name+'.png'))
    plt.savefig(os.path.join(file_images, time.strftime("%Y-%m-%d-%H-%M") + 'TSP vs discount' +str(n) + file_name+'.png'))
    plt.show()

    plt.title(str(n) +" customers, " + file_name)
    plt.scatter(tsp_cost_x, num_discounts_y, c="red", label = "TSP")
    plt.plot(tsp_cost_x, num_discounts_y, c="red")
    plt.scatter(tsp_and_discount_cost_x, num_discounts_y, c="blue", label = "TSP + Expected Discount")
    plt.plot(tsp_and_discount_cost_x, num_discounts_y, c="blue")
    plt.ylabel('min # of discounts')
    plt.xlabel('Cost (worst case)')
    plt.legend()
    plt.savefig(os.path.join(file_images, time.strftime("%Y-%m-%d-%H-%M") + 'TSP vs discount size' +str(n) + '.png'))
    plt.show()
    plt.figure(2)
    ax = plt.gca()
    fig = plt.gcf()
    constrained_layout = True
    ax.set_aspect(1.0)
    fig, ax = print_vertex(input_data, Y[0], fig, ax, file_name, sets["N"])
    plt.show()

    opt_model = grb.Model(name="MIP Model")
    opt_model.setParam('OutputFlag', False)
    opt_model.setParam('Threads', constants.LIMIT_CORES)
    teta_vars = {y: opt_model.addVar(vtype=grb.GRB.BINARY,
                                     name="y_{0}".format(y)) for y in Y}


    opt_model.addConstr(grb.quicksum(teta_vars[y] for y in Y) == 1)

    objective = 0
    for y in Y:
        objective += teta_vars[y] * sum(prob_matrix[y][w] * C[w] for w in W)
        for i in sets['N']:
            objective += teta_vars[y] * set_customer_probability(Y[y][i]*discount, input_data) * Y[y][i]*discount


    opt_model.update()
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)
    opt_model.optimize()

    for y in Y:
        if teta_vars[y].x > 0.1:
            opt_strategy_stochastic = Y[y]
    # print("optimal", opt_strategy_stochastic)
    expected_cost_stoch = opt_model.objVal

    print("optimal stochastic strategy", opt_strategy_stochastic, round(expected_cost_stoch,2))
    for reduction in [ 0, 0.025, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5]:
        for y in dict_allowed_y[reduction]:
            exp_data = {}

            disc = 0
            for i in sets["N"]:
                if y[i] == 1:
                    disc += 1
            if disc == min_disc_size[reduction]:
                exp_cost = 0
                for w in W:
                    for i in Y:
                        if Y[i] == y:
                            num_y = i
                    exp_cost += C[w]*prob_matrix[num_y][w]
                exp_data['Alpha(reduction)'] = reduction
                exp_data['Policy'] = format_strategy(y)
                exp_data['Exp_cost'] =  round(exp_cost,2)
                exp_data['#_discounts'] = disc

                write_table(file_results, exp_data, head)
    print(instances_num)


    #for i in range(n+1):
    #    x.append(i)
    #    weight.append((0.8**(n-i)*0.2**i)*math.factorial(n) / (math.factorial(n - i) * math.factorial(i)))

    #plt.figure(1)
    #plt.ylabel('Weight')
    #plt.xlabel('Number of deviated customers')
    #plt.title('Weight of all scenarios that deviate for strategy in x customers, n = 100')
    #plt.plot(x,weight)
    #plt.plot(x, y2)
    #plt.savefig(' TSP_costs.png')
    #plt.show()

   # for i in range(0, 7):
   #     cum += y[i]
   #     print(i, y[i])










