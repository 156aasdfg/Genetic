import numpy as np
import random
import math
import matplotlib.pyplot as plt


POP_SIZE = 30
DNA_SIZE = 10
CROSS_RATE = 0.5         
MUTATION_RATE = 0.005    
N_GENERATIONS = 30

def function(x):#设定目标函数  Set the aim function
    return 5*np.sin(x)+10*np.cos(5*x)

def fitness(pred):#设定适应度函数  Set fitness function
    if pred < 0:#适应度函数不为0  The fitness function should not equal to zero.
        pred = 0
    return pred + 1e-3

def decoding(pop, DNA_SIZE):#解码过程，将二进制转变为十进制 Decoding process, converting binary to decimal
    pop_copy=[]
    for i in range(len(pop)):
        sum = 0
        for j in range(DNA_SIZE):
            sum+=pop[i][j]* (2 ** (DNA_SIZE - 1 - j))
        pop_copy.append(sum)
    return np.array(pop_copy)

def decoding4only(maxfit,DNA_SIZE):#Decoding process for single sample
    for i in range(DNA_SIZE):
        a=maxfit[i]* (2 ** (DNA_SIZE - 1 - i))
    return a

def selection(pop,fitness):#轮赌选择  Roulette selection

    fitness_sum=[]
    for i in range(len(fitness)):
        if i ==0:
            fitness_sum.append(fitness[i])
        else:
            fitness_sum.append(fitness_sum[i-1]+fitness[i])

    for i in range(len(fitness_sum)):
        fitness_sum[i]/=sum(fitness)
    #轮赌概率 Roulette probability 
    pop_new=[]
    for i in range(len(fitness)):
        rand = np.random.uniform(0,1)
        for j in range(len(fitness)):
            if j==0:
                if 0<rand and rand<=fitness_sum[j]:
                    pop_new.append(pop[j])

            else:
                if fitness_sum[j-1]<rand and rand<=fitness_sum[j]:
                    pop_new.append(pop[j])             
    return np.array(pop_new)


def crossover(parent, pop):#交叉 
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   
        pop = np.array(pop)
        parent[cross_points] = pop[i_, cross_points]                           
    return parent

def mutate(child):#变异
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))

y = []

for _ in range(N_GENERATIONS):
    F_values = function(decoding(pop, DNA_SIZE))
    fitness = fitness(F_values)
    maxfit = []
    maxfit = pop[np.argmax(fitness), :]
    print("Most fitted DNA: ", maxfit)
    print("Most Value:", function(decoding4only(maxfit, DNA_SIZE)))
    y.append(function(decoding4only(maxfit, DNA_SIZE)))
    pop = selection(pop, fitness)
    pop_cp = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_cp)
        child = mutate(child)
        parent[:] = child

x1 = range(0, N_GENERATIONS)
plt.subplot(1, 1, 1)
plt.plot(x1, y, 'o-')
plt.xlabel('N_GENERATIONS')
plt.ylabel('Value')
plt.show()
plt.savefig('GA.png')