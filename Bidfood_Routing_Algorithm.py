import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from docplex.mp.model import Model
import time

#True for saving plot, False otherwise
save_fig = False

#5, 7, 10, 12, 14 customers db exist
cust_excel = 5

'''read the customer data from the excel file'''
data_path = (r'C:\Users\VRP%s' % (cust_excel) + '.xlsx')
data_path = (r'C:VRP%s' % (cust_excel) + '.xlsx')
df = pd.read_excel(data_path, engine="openpyxl")
n = df.shape[0] - 1  # number of customers

'''FIXED PARAMETERS SECTION'''
#########################################################################################
Hv = 8*60 # 8 hours
Speed = 25 # speed km/h divide by 4.6 --> in m/s
M = 1000 # big number


LocationsCustomers = [i for i in range(1, n + 1)] # Lc --> set locations without depot
Locations = [0] + LocationsCustomers #+ [n+1]# L --> set locations with depot

Products = [1, 2] # Set of product categories

Vehicles = ["A", "B", "C", "D", "E", "F", "G", "H"]
Wmax = 7490 # MAX vehicle load # or. val. =7490
Wempt = 3240 #kg empty weight
MaxLoad = 4250 #capacity weight is 4250KG
Qv_fresh = 9 # capcity rollcontainers = 12
Qv_frozen = 3
#Qv = Qv_fresh + Qv_frozen
Bv = 144 # battery capacity in kWh

alpha = 0.0981
beta = 2.107175
EngineEff = 0.8

'''VARIABLE PARAMETERS SECTION'''
########################################################################################
loc_x = df["x"] # coordinate x for customer i
loc_y = df["y"] # coordinate y for customer i
DemandFresh = df['d1'] # demand of product fresh customer i
DemandFrozen = df['d2'] # demand of product frozen customer i
DemandTotal = df['dt'] # demand of product frozen customer i
DemandWeight = df["dw"] # demand weight customer i
et = df["et"] # earliest time start customer i
lt = df["lt"] # latest time start customer i
ut = df["ut"] # service time at customer i


mdl = Model('MCVRPTWLD') # CREATE CPLEX MODEL

'''DECISION VARIABLES SECTION'''
#########################################################################################
A = [(i, j, v) for i in Locations for j in Locations for v in Vehicles] # set of arcs / vertices (i, j)
B = [(i, v) for i in Locations for v in Vehicles] # create empty array for variables with i,v
D = [(p, v) for p in Products for v in Vehicles]  # create empty array for variables with p, v
E = [(i, p) for i in LocationsCustomers for p in Products]  # create empty array for variables with i, p


x = mdl.binary_var_dict(A, name = 'x') # xjiv --> if customer i is served by vehicle (v)
z = mdl.continuous_var_dict(B, name = 'z', ub=Bv) # ziv --> remaining battery charge of vehicle v on arrival at j from i
b = mdl.continuous_var_dict(B, name = 'b') # biv -->arrival time at customer i
#Q = mdl.integer_var_dict(D, name = 'Q') # Qpv --> capacity units of product type p of vehicle (v)
W = mdl.continuous_var_dict(B, name = 'W', lb=0, ub=MaxLoad) # wijv --> vehicle load (KG) on arc (i, j) for vehicle (v)
J = mdl.continuous_var_dict(A, name = 'J', lb=0) # Jijv --> Energy consumption on arc (i, j) for vehicle (v)

CumEn = mdl.continuous_var_dict(B, name = 'CumEn', ub=Bv)

Qfresh = mdl.continuous_var_dict(B, name='Qfresh', ub=Qv_fresh) #cumulative demand
Qfrozen = mdl.continuous_var_dict(B, name='Qfrozen', ub=Qv_frozen) #cumulative demand


# DISTANCE AND TIME FUNCTION CALCULATION
a = {(i, j): round(np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j])) for i in Locations for j in Locations} # calculate euclidian distance on arc (i, j)
t = {(i, j): round(((np.hypot(loc_x[i] - loc_x[j], loc_y[i] - loc_y[j])) / Speed * 60)) for i in Locations for j in Locations} # calculate travel time over arc (i, j)

print(a)
'''CONSTRAINTS'''
#########################################################################################
'''BASIC ROUTE'''
#constraint 2

for i in LocationsCustomers:
    mdl.add(mdl.sum(x[i, j, v] for v in Vehicles for j in Locations) == 1) # Each point must be visited exactly once

#constraint 3 & 4
for v in Vehicles:
    mdl.add(mdl.sum(x[0, j, v] for j in LocationsCustomers) <= 1)
    mdl.add(mdl.sum(x[i, 0, v] for i in LocationsCustomers) <= 1)
    #mdl.add(mdl.sum(x[i, n + 1, v] for i in LocationsCustomers if i != n + 1) == 1 for v in Vehicles)
#constraint 5
for i in LocationsCustomers:
    for v in Vehicles:
        mdl.add(mdl.sum(x[i, j, v] for j in Locations if i != j) == mdl.sum(x[j, i, v] for j in Locations if i != j)) # route continuity: ensure that if a vehicle arrives at location i, it also leaves

'''DEMAND & CAPACITY'''
#constraint 6
for v in Vehicles:
    mdl.add(Qfresh[i, v] >= 0)
    mdl.add(Qfrozen[i, v] >= 0)
    mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], Qfresh[i, v] + (DemandFresh[j]) == Qfresh[j, v]) for i in Locations for j in Locations if i != 0 and j != 0)
    mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], Qfrozen[i, v] + (DemandFrozen[j]) == Qfrozen[j, v]) for i in Locations for j in Locations if i != 0 and j != 0)
    mdl.add(Qfresh[i, v] >= (DemandFresh[i]) for i in LocationsCustomers)
    mdl.add(Qfrozen[i, v] >= (DemandFrozen[i]) for i in LocationsCustomers)

'''WEIGHT TRACKING OF VEHICLE'''
#constraint 7 tracking vehicle load
for i in Locations:
    for v in Vehicles:
        mdl.add(W[0, v] == 0 for i in Locations)
        mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], W[i, v] - DemandWeight[i] >= W[j, v]) for i in Locations for j in Locations for v in Vehicles if i != 0 and j != 0)
        mdl.add(W[i, v] >= DemandWeight[i] for i in Locations)

'''ENERGY CONSTRAINTS'''
#constraint 8 energy use
for v in Vehicles:
    mdl.add((J[i, j, v] == ((((alpha * (Wempt + W[j, v]) * a[i, j]) + (beta * (4.1666667 ** 2) * a[i, j])) / EngineEff) / 500)) for i in Locations for j in Locations for v in Vehicles)

for i in Locations:
    for v in Vehicles:
        mdl.add(CumEn[i, v] >= 0 for i in Locations for j in Locations)
        mdl.add(CumEn[i, v] <= Bv for i in Locations for j in Locations)

        #Cumulative energy usage function
        mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], CumEn[i,v] + J[i,j,v] == CumEn[j,v]) for i in Locations for j in Locations for v in Vehicles if j != 0)
        mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], Bv - CumEn[i,v] >= CumEn[j,v]) for i in Locations for j in Locations for v in Vehicles if j != 0)

        #Remaining battery function
        mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], Bv - CumEn[i,v] == z[i, v]) for i in Locations for j in Locations for v in Vehicles)



'''TIME CONSTRAINTS'''
#constraint 13, max driver time per day
for v in Vehicles:
    mdl.add(mdl.sum(x[i, j, v] * t[i, j] + ut[i] for i in Locations for j in Locations) <= 100000)

#constraint 14, time windows
    mdl.add(b[0, v] == 6 for v in Vehicles)
    mdl.add_constraints(b[i, v] >= et[i] for i in LocationsCustomers for v in Vehicles)
    mdl.add_constraints(b[i, v] <= lt[i] for i in LocationsCustomers for v in Vehicles)
    mdl.add_indicator_constraints(mdl.indicator_constraint(x[i, j, v], (b[i, v] + (ut[i]/60) + t[i, j]) <= b[j, v]) for i in Locations for j in Locations for v in Vehicles if j != 0 if i!=n+1)

'''TIME LIMIT (E.G. SAVING VM COSTS)'''
mdl.parameters.timelimit = 15 # Add running time limit

'''OBJECTIVE FUNCTION'''
obj_function = (mdl.sum(J[i, j, v] * x[i, j, v] for v in Vehicles for i in Locations for j in Locations if i != j))
#obj_function = (mdl.sum(a[i,j] * x[i, j, v] for i in Locations for j in Locations for v in Vehicles))

# Solve
mdl.minimize(obj_function)
mdl.parameters.timelimit = 15
solution = mdl.solve(log_output=True)


if solution:
    #arc_weight = mdl.solution.get_values(W[i,v] for i,v in B)
    #arc_traversed = mdl.solution.get_values(x[i,j,v] for i,j,v in A)
    #arc_energy = mdl.solution.get_values(J[i,j,v] for i,j,v in A)
    #vehic_demand = mdl.solution.get_values(Q[i, v] for i, v in B)
    cum_en = mdl.solution.get_values(CumEn[i, v] for i, v in B)
    print(solution.solve_status, '\n') # Returns if the solution is Optimal or just Feasible
    print(solution)
    #route = [y[0, i, v] for i in LocationsCustomers for v in A if y[0, i, v].solution_value == 1]
    #no_vehicles = len(route)
    #print(no_vehicles)
    #print(mdl.export_to_string())
else:
    print('Not feasible')


active_arcs = [k for k in A if x[k].solution_value > 0.9]
#print(np.array(active_arcs))


Route_A = []
Route_B = []
Route_C = []
Route_D = []
Route_E = []
Route_F = []
Route_G = []
Route_H = []

for i,j,v in active_arcs:
    if v == "A":
        Route_A.append([i,j])
    if v == "B":
        Route_B.append([i,j])
    if v == "C":
        Route_C.append([i,j])
    if v == "D":
        Route_D.append([i,j])
    if v == "E":
        Route_E.append([i,j])
    if v == "F":
        Route_F.append([i,j])
    if v == "G":
        Route_G.append([i,j])
    if v == "H":
        Route_H.append([i,j])

print("Route A: ", Route_A, "\nRoute B: ", Route_B, "\nRoute C: ", Route_C, "\n", "\nRoute D: ",Route_D, "\n", "\nRoute E: ",Route_E, "\n", "\nRoute F: ",Route_F, "\n", "\nRoute G: ",Route_G, "\n", "\nRoute H: ",Route_H, "\n")

for i, j, v in active_arcs:
    #output = ("Node %s -> %s"  % arcs + ",  dist= %s" % a[arcs])
    output = ("Node %d -> %d by vehicle %s, dis= %s, time= %s" %(i, j, v, a[i,j], t[i,j]))
    #output = (arcs, dis[arcs])
    print(output)

#PLOTTING THE ROUTES IN A GRAPH
#########################################################################################
# plotting the routes
fig = plt.figure(figsize = (5, 5), dpi = 200)
plt.scatter(loc_x[1:], loc_y[1:], color='black', label='Customers', marker='.')
plt.xlabel("X-coordinates")
plt.ylabel("Y-coordinates")
plt.title("MCVRPTWLD routes")
plot_line = 0.7 #line thickness
plot_alpha = 1 #line transparancy  0-1


for i, j, v in active_arcs:
    if v == "A":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='mediumblue', linewidth=plot_line, alpha=plot_alpha)
    if v == "B":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='darkgreen', linewidth=plot_line, alpha=plot_alpha)
    if v == "C":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='cyan', linewidth=plot_line, alpha=plot_alpha)
    if v == "D":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='darkgoldenrod', linewidth=plot_line, alpha=plot_alpha)
    if v == "E":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='fuchsia', linewidth=plot_line, alpha=plot_alpha)
    if v == "F":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='darkgray', linewidth=plot_line, alpha=plot_alpha)
    if v == "G":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='orange', linewidth=plot_line, alpha=plot_alpha)
    if v == "H":
        plt.plot([loc_x[i], loc_x[j]], [loc_y[i], loc_y[j]], color='red', linewidth=plot_line, alpha=plot_alpha)
'''
for i in LocationsCustomers:
    plt.annotate('$Customer={%s}$\n$Q={%s } _& { %s}$'%(i, DemandFresh[i], DemandFrozen[i]), (loc_x[i]-1.2, loc_y[i]+1), fontsize=3, color='black')
'''
plt.plot(loc_x[0], loc_y[0], marker='s', color='firebrick', label='Depot', ms=7)

plt.legend()
plt.show()

if save_fig == True:
    fig.savefig('plot.png', bbox_inches='tight')
