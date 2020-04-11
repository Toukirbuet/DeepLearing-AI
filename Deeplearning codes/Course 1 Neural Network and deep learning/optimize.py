# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:57:57 2020

@author: Real
"""
from propagate import propagate

# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
    # Cost and gradient calculation ( 1-4 lines of code)
    ### START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        # update rule ( 2 lines of code)
        ### START CODE HERE ###
        w = w-learning_rate*dw
        b = b-learning_rate*db
        ### END CODE HERE ###
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training examples
        #if print_cost and i % 100 == 0:
           # print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,"b": b}
    grads = {"dw": dw,"db": db}
    return params, grads, costs