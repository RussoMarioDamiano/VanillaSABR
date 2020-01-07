import numpy as np

def DifferentialEvolution(object_function, bounds, args=(),
                          mut=0.8, crossp=0.7, popsize=20, epochs=2000,
                          method="resample"):
    """
    Agent-based constrained optimization algorithm.
    Returns a list of N=`epochs` two-element tuples in the form:
        (np.array, float)
    The first element of the tuple contains the estimated parameters,
    the second one contains the value of the loss function.
    
    Freely adapted from https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
    
    Parameters
    ----------
    object_function : function
        The function in K parameters to minimize.
        The object function must:
            - Have a list of K parameters as first input
            - Have all the other arguments (if any) come after the
              list of K parameters. These arguments are passed 
              to the object function through the `args` argument.
              e.g.: object_functions(params, arg1, arg2, ...)
    bounds : iterable
        List or array of length K mad of 2-element tuples (lower_bound, upper_bound). 
        The tuples must be ordered according to the input order of object_function.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` and `hess` functions).
    mut : float in [0, 2], optional
        Rate of mutation of agents.
        If higher, there will be more diversity in the offspring.
        Higher values lead to better parameter estimates, but slow down convergence.
    crossp : float, optional
        Probability of replacement of features in the agent with feature from the mutated individual.
    popsize : int, optional
        Number of agents to initialize.
        Higher number of agents leads to better estimates, but slows down convergence.
    epochs : int, optional
        Number of training epochs.
    method : ["resample", "clip"], optional
        Method of mutant coefficient shrinkage when outside [0,1].
        "resample" resamples from a uniform(0,1).
        "clip" clips to the closest value between 0 and 1.
    """
    
    # number of parameters we need to estimate
    dimensions = len(bounds)
    
    # INITIALIZE AGENTS
    # initialize agents to a random state
    # each agent will be a list of K randomly-initialized values
    pop = np.random.rand(popsize, dimensions)
    
    # obtain the two arrays of upper bounds and lower bounds, both of length K
    min_b, max_b = np.array(bounds).T
    # scale values from [0, 1] to [min, max] (denormalization)
    # > the idea is to get the "distance" between any two bounds...
    diff = np.fabs(min_b - max_b)
    # > ...then multiply the values in [0,1] by that "distance" and add/subtract the min/max value.
    pop_denorm = pop * diff + min_b # = pop * diff - max_b
    
    # store the agents' performances in an array
    fitness = np.asarray([object_function(ind, *args) for ind in pop_denorm])
    
    # store the index of the best agent ...
    best_idx = np.argmin(fitness)
    # ... and the best performing agent
    best = pop_denorm[best_idx]
    
    for epoch in range(epochs):
        # for each agent `j` in the population:
        for j in range(popsize):
            # pop[j] is now our "target" agent
            # get the indices of all the other agents EXCEPT the target `j`
            idxs = [idx for idx in range(popsize) if idx != j]

            # choose 3 agents from the NORMALIZED pop *without* replacement.
            selected = np.random.choice(idxs, 3, replace=False)
            a, b, c = pop[selected]

            # create a mutant vector by combining a, b, c:
            # add to `a` the difference between b and c multiplied by the mutation constant
            mutant = a + mut * (b-c)
            
            # mutant might have values above 1 or below 0, so you have to bring them inside the boundaries:
            if method == "resample":
                # either extract new random numbers in [0, 1] if elements are outside the boundaries
                mutant = np.where(((mutant<0)|(mutant>1)), np.random.rand(dimensions), mutant)
            elif method == "clip":
                # or clip the values outside the boundaries to either 0 or 1.
                mutant = np.clip(mutant, a_min=0, a_max=1)
            
            # RECOMBINATION PHASE
            # have the mutant "mix" with the target agent
            # determine if an element in target needs to be replaced with the element in mutant at the same position
            # based on the value `crossp`. `crossp` represents the "percentage" of mutant that will be kept
            # > create a boolean vector of size (1 X dimensions)
            cross_points = np.random.rand(dimensions) < crossp
            # where cross_points == True, replace target with mutant to create `trial`
            trial = np.where(cross_points, mutant, pop[j])

            # now we need to de-normalize `trial`
            trial_denorm = trial * diff + min_b

            # lastly, test trial_denorm's performance against target.
            # if it's better, replace target with trial_denorm.
            # this ensures that future recombinationss will be conducted with more fit individuals
            fitness_trial = object_function(trial_denorm, *args)
            if fitness_trial < fitness[j]:
                fitness[j] = fitness_trial
                pop[j] = trial
                # lastly, if trial's performance is better than the best performer, store that as well.
                if fitness_trial < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
            
        yield best, fitness[best_idx]