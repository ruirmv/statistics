import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

def markov_mc_simulation(N, distribution, test):
    """
    Computes the result of a Markov simulation with N samples from a given 
    distribution. 
    
    The test should return True/False. The test method will be called by 
    unpacking the return value of distribution.
    
    """
    positives = 0
    for i in range(N):
        result = test(*distribution())
        if result: positives += 1
        
    return positives/N

def pi_markov_mc(N):
    """
    Computes an approximate value for pi using a Markov simulation with N samples
    """
    return 4 * markov_mc_simulation(N, lambda: (rd.random_sample(), rd.random_sample()), 
                                 lambda x, y : x**2 + y**2 < 1)
    
def hist_pi_markov_mc(N_sim, N, n_bins=50):
    """
    Displays an histogram of the result of N_sim Markov simulations for 
    computing pi, each of which with N samples drawn. Returns the mean value 
    and variance obtained
    """
    aux = [pi_markov_mc(N) for x in range(N_sim)]
    plt.hist(aux, n_bins)
    return np.mean(aux), np.var(aux)

if __name__ == "__main__":
    hist_pi_markov(1000, 100, 50)
