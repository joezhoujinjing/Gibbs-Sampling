# Gibbs sampling algorithm to denoise an image
# author : Jinjing Zhou, Gunaa AV, Isaac Caswell
# date : Feb 1st, 2016

import math
import copy
import random
from sys import stdout
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

MAX_BURNS = 100
MAX_SAMPLES = 1000
CACHE={}
#eta = 1, beta = 1, B = 100, and S = 1000
def markov_blanket(i,j,Y,X):
    '''
    return: 
        the a list of Y values that are markov blanket of Y[i][j]
        e.g. if i = j = 1, 
            the function should return [Y[0][1], Y[1][0], Y[1][2], Y[2][1], X[1][1]]
    '''
    return (Y[i+1][j],Y[i-1][j],Y[i][j+1],Y[i][j-1],X[i][j])

def sampling_prob(markov_blanket):
    '''
    markov_blanket: a list of the values of a variable's Markov blanket
        The order doesn't matter (see part (a)). e.g. [1,1,-1,1]
    return:
         a real value which is the probability of a variable being 1 given its Markov blanket
    '''
    if markov_blanket in CACHE:
        return CACHE[markov_blanket]
    s=sum(markov_blanket)
    res=1.0/(1+math.exp(-2*s))
    CACHE[markov_blanket]=res
    return res


def sample(i, j, Y, X, DUMB_SAMPLE = 0):
    '''
    return a new sampled value of Y[i][j]
    It should be sampled by 
        (i) the probability condition on all the other variables if DUMB_SAMPLE = 0
        (ii) the consensus of Markov blanket if DUMB_SAMPLE = 1
    '''
    blanket = markov_blanket(i,j,Y,X)
    if not DUMB_SAMPLE:
        prob = sampling_prob(blanket)
        if random.random() < prob:
            return 1
        else:
            return -1
    else:
        c_b = Counter(blanket)
        if c_b[1] >= c_b[-1]:
            return 1
        else: 
            return -1

def gibbs_sample(Y, X, DUMB_SAMPLE = 0):
    '''
    performaing Gibbs sampling on the whole Y
    (one new sample for each entry of Y)
    '''
    for i in xrange(1,len(Y)-1):
        for j in xrange(1,len(Y[0])-1):
            Y[i][j] = sample(i,j,Y,X,DUMB_SAMPLE)
    return copy.deepcopy(Y)



def get_posterior_by_sampling(filename, Y, DUMB_SAMPLE = 0):
    '''
    Do Gibbs sampling on the image specified in filename.
    If not dumb_sample, it should run MAX_BURNS iterations of burn in and then
    MAX_SAMPLES iterations for collecting samples.
    If dumb_sample, run MAX_SAMPLES iterations and returns the final image.

    filename: file name of image in txt
    logfile: the file name that stores the energy log (will use for plotting later)
        look at the explanation of plot_energy to see detail
    DUMB_SAMPLE: equals 1 if we want to use the trivial reconstruction 

    return value: posterior, Y, frequencyZ
        posterior: an 2d-array with the same size of Y, the value of each entry should
            be the probability of that being 1 (estimated by the Gibbs sampler)
        Y: The final image (for DUMB_SAMPLE = 1, in part (d))
        frequencyZ: a dictionary with key: count the number of 1's in the Z region
                                      value: frequency of such count
    '''
    X=read_txt_file(filename)
        
    if DUMB_SAMPLE==1:
        Y=copy.deepcopy(X)
        plt.subplot(131),plt.imshow(read_txt_file('orig.txt'),cmap='Greys'),plt.title('ORIGINAL')
        plt.subplot(132),plt.imshow(X,cmap='Greys'),plt.title('OBSERVED')
        for i in range(30):
            stdout.write("\r%d percent" %(i*3))
            stdout.flush()
            gibbs_sample(Y,X,DUMB_SAMPLE=DUMB_SAMPLE)
        plt.subplot(133),plt.imshow(Y,cmap='Greys'),plt.title('RECOVERED')
        Original=read_txt_file('orig.txt')
        total=0
        DScorrect=0
        for i in xrange(1,len(Y)-1):
                for j in xrange(1,len(Y[0])-1):
                    if Y[i][j]==Original[i][j]:
                        DScorrect+=1
                    total+=1
        print 'DS Correction rate: %s'%(1.0*DScorrect/total)
        plt.show()
        return Y

    else:
        newY=[[0]*(len(X[0])) for i in xrange(len(X))]
        for i in xrange(MAX_SAMPLES):
            if i%(MAX_SAMPLES/100)==0:
                    x=i/10+1
                    stdout.write("\r%d percent" %x)
                    stdout.flush()
            y=gibbs_sample(Y,X)
            #Posterior probability
            for m in xrange(1,len(newY)-1):
                for n in xrange(1,len(newY[0])-1):
                    newY[m][n]+=y[m][n]
        for i in xrange(len(newY)):
                for j in xrange(len(newY[0])):
                    newY[i][j]=1 if newY[i][j]>0 else -1
        plt.subplot(131),plt.imshow(read_txt_file('orig.txt'),cmap='Greys'),plt.title('ORIGINAL')
        plt.subplot(132),plt.imshow(X,cmap='Greys'),plt.title('OBSERVED')
        plt.subplot(133),plt.imshow(newY,cmap='Greys'),plt.title('RECOVERED')

        Original=read_txt_file('orig.txt')
        total=0
        GScorrect=0
        for i in xrange(1,len(newY)-1):
                for j in xrange(1,len(newY[0])-1):
                    if newY[i][j]==Original[i][j]:
                        GScorrect+=1
                    total+=1
        print 'GS Correction rate: %s'%(1.0*GScorrect/total)
        plt.show()

def plot_energy(filename): 
    '''
    filename: a file with energy log, each row should have three terms separated by a \t:
        iteration: iteration number
        energy: the energy at this iteration
        S or B: indicates whether it's burning in or a sample
    e.g.
        1   -202086.0   B
        2   -210446.0   B
        ...
    '''
    its_burn, energies_burn = [], []
    its_sample, energies_sample = [], []    
    with open(filename, 'r') as f:
        for line in f:
            it, en, phase = line.strip().split()
            if phase == 'B':
                its_burn.append(it)
                energies_burn.append(en)
            elif phase == 'S':
                its_sample.append(it)
                energies_sample.append(en)                
            else:
                print "bad phase: -%s-"%phase

    p1, = plt.plot(its_burn, energies_burn, 'r')
    p2, = plt.plot(its_sample, energies_sample, 'b')    
    plt.title(filename)
    plt.legend([p1, p2], ["burn in", "sampling"])
    plt.show()

def read_txt_file(filename):
    '''
    filename: image filename in txt
    return:   2-d array image
    '''
    f = open(filename, "r")
    lines = f.readlines()
    height = int(lines[0].split()[1].split("=")[1])
    width = int(lines[0].split()[2].split("=")[1])
    Y = [[0]*(width+2) for i in range(height+2)]
    for line in lines[2:]:
        i,j,val = [int(entry) for entry in line.split()]
        Y[i+1][j+1] = val
    return Y

def calculate_energy(Y,X):
    res=0
    for i in xrange(1,len(Y)-1):
        for j in xrange(1,len(Y[0])-1):
            blanket=markov_blanket(i,j,Y,X)
            res+=Y[i][j]*(blanket[-1]+0.5*sum(blanket[:-1]))
    return -res



def burn_in_choice(filename,logfile):
    burn_in=-1
    energy_record=[0]*30
    X=read_txt_file('noisy_20.txt')
    Y=[[1 if random.random()<0.5 else -1]*(len(X[0])) for i in xrange(len(X))]
    f=open(logfile,'w')
    for i in range(MAX_SAMPLES):
        gibbs_sample(Y,X)
        l='BS'[i>=MAX_BURNS]
        energy=calculate_energy(Y,X)
        energy_record[:-1]=energy_record[1:]
        energy_record[-1]=energy
        f.write('%s  %s  %s\n'%(i,energy,l))
        if i%(MAX_SAMPLES/100)==0:
            x=i/10+1
            stdout.write("\r%d percent and energy: %d" % (x,energy))
        if (abs(energy_record[-1])*30-abs(sum(energy_record)))/abs(energy_record[-1])<0.01:
            if burn_in<0:
                burn_in=i
                print 
                print '-'*30,'it is a valid burn-in number.','-'*30
                return i,Y
    #plot energy graph if burn-in is too short.
    plot_energy(logfile)
    return burn_in

    f.close()
    plot_energy(logfile)

if __name__ == '__main__':
    print 
    print 'We first look at the naive method to recover a picture, the pixel will choose the values that majority of its neigbour choose.'
    get_posterior_by_sampling('noisy_20.txt',None,DUMB_SAMPLE = 1)
    print 'For Gibbs Sampling method, we first evaluate the validity of burn-in number...'
    conv,Y=burn_in_choice('noisy_20.txt','plot_energy')
    assert conv<MAX_BURNS
    stdout.write('The samples\' energy level converge within the burn-in number of %s'%conv)
    get_posterior_by_sampling('noisy_20.txt',Y,DUMB_SAMPLE = 0)
