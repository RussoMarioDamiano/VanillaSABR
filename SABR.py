import numpy as np
from DifferentialEvolution import DifferentialEvolution
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt

def SABR(alpha, beta, rho, nu, F, K, time, MKT=0):
    """
    Returns estimated market volatilities using Hagan's [2002] formulae.
    Uses the shortened form in the ATM case to speed up computations.
    Each input value is a scalar.
    """

    if K <= 0:
        raise Exception("Negative Rates Detected. Need to shift the smile.")

    elif F == K: #ATM shortened formula
        V = (F*K)**((1-beta)/2)
        B = 1 + (((1-beta)**2 * alpha**2)/(24*(V**2)) + (rho*beta*alpha*nu)/(4*V) + ((2-3*(rho**2))*(nu**2))/(24)) * time
        # estimated volatility
        VOL = (alpha/V)*B

    elif F != K: #NON-ATM formula
        V = (F*K)**((1-beta)/2)
        log_FK = np.log(F/K)
        z = (nu*V*log_FK)/alpha
        x = np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho))
        A = 1 + ((1-beta)**2*log_FK**2)/24 + ((1-beta)**4*log_FK**4)/1920
        B = 1 + (((1-beta)**2 * alpha**2)/(24*(V**2)) + (rho*beta*alpha*nu)/(4*V) + ((2-3*(rho**2))*(nu**2))/(24))*time
        # estimated volatility
        VOL = (alpha/(V*A)) * (z/x) * B

    return VOL





def objfunc(par, beta, F, K, time, MKT):
    """
    Sum of Square Residuals (SSR) function for the SABR model.
    It is used as an input in the calibration phase.
    The goal of calibration is to minimize the output of this function.
    """

    alpha, rho, nu = par

    sum_sq_diff = 0
    for j in range(len(K)):
        if MKT[j] == 0: #we interpret 0 as missing data - very unlikely that market volatility is 0!!
            diff = 0
        elif F == K[j]: #ATM shortened formula
            V = (F*K[j])**((1-beta)/2)
            B = 1 + (((1-beta)**2 * alpha**2)/(24*(V**2)) + (rho*beta*alpha*nu)/(4*V) + ((2-3*(rho**2))*(nu**2))/(24)) * time
            VOL = (alpha/V)*B
            diff = VOL - MKT[j]
        elif F != K[j]: #NON-ATM formula
            V = (F*K[j])**((1-beta)/2)
            log_FK = np.log(F/K[j])
            z = (nu*V*log_FK)/alpha
            x = np.log((np.sqrt(1-2*rho*z+z**2)+z-rho)/(1-rho))
            A = 1 + ((1-beta)**2*log_FK**2)/24 + ((1-beta)**4*log_FK**4)/1920
            B = 1 + (((1-beta)**2 * alpha**2)/(24*(V**2)) + (rho*beta*alpha*nu)/(4*V) + ((2-3*(rho**2))*(nu**2))/(24)) * time
            VOL = (alpha/(V*A)) * (z/x) * B
            diff = VOL - MKT[j]
        sum_sq_diff += diff**2
    return sum_sq_diff






class SABR_swaption():
    """
    Calibrate a SABR model on given market data.
    Parameters estimation is done via Differential Evolution.

    Parameters
    ----------
    F : float
        The forward rate of the swaption
    K : iterable
        Iterable of length N containing the strike prices on which the model will be calibrated.
    time : float
        Time to maturity (in years) of the swaption.
    vols : iterable
        Iterable of length N containing the observed market volatilities.
    beta : float
        Value of the beta parameter of the SABR model.
        It is chosen a priori, usually in (0, 0.5, 1).
    s : float
        Imposed lower bound for strike rates when one of the strike rates is below 0.
        When this happens, a standard practice is to "shift" rates to make them > 0.
    calibration : ["SLS_SciPy", "DE_SciPy", "DE_homebrew"], optional
        Algorithm to use for parameter calibration.
        "SLS_SciPy" refers to SciPy's implementation of Sequential Least Squares [recommended].
        "DE_SciPy" refers to SciPy's implementation of DifferentialEvolution.
        "DE_homebrew" refers to the attached DifferentialEvolution code made by git:russomariodamiano.
        Outcomes will be the same, but Scipy's code is optimized and thus faster.
    starting_params : iterable, optional
        [Not used in Differential Evolution calibration]
        Initializes parameters for SLS calibration.
        Input order is: [alpha, rho, nu]
        Proper input can speed up convergence.
    """
    def __init__(self, F, K, time, vols, beta=0.5,
                 s=0.001, calibration="SLS_SciPy",
                 starting_params=[0.001, 0, 0.001]):

        try:
            assert len(vols) == len(K)
        except AssertionError:
            raise Exception("`K` and `vols` must have the same length")

        self.F = F
        self.K = np.array(K)
        self.time = time
        self.vols = np.array(vols)
        self.beta = beta
        self.s = s

        # check that no strike is below 0, else shift it
        if np.any(self.K < 0):
            self.shift_smile()
        else:
            self.shift = 0 # for plotting-adjustement purposes

        # compute the curve parameters
        self.calibration(calibration, starting_params)
        
        # after calibration is done, un-shift the smile
        self.K -= self.shift


    def shift_smile(self):
        """
        The SABR model from Hagan's 2002 paper relies on the assumption that strikes are positive.
        In the post-2008 market, rates can often be negative, and so the model needs to be corrected.
        The shifted SABR simply shifts the smile to the right so that strikes are lower-bounded by an arbitrary value, `s`.
        """
        self.shift = self.s - np.min(self.K)
        self.K += self.shift
        self.F += self.shift


    def calibration(self, calibration, starting_params):
        """
        Calibrate the SABR model on the provided volatility data.
        Calibration is done either via:
            - Sequential Least Squares
            - Differential Evolution
        """

        # Calibration using SciPy's Sequential least Squares
        if calibration == "SLS_SciPy":
            bnds = ( (0.001,None) , (-0.999,0.999) , (0.001,None)  )
            res = minimize(objfunc, x0=starting_params,
                           args=(self.beta, self.F, self.K, self.time, self.vols),
                           bounds = bnds, method='SLSQP')

            self.alpha = res.x[0]
            self.rho = res.x[1]
            self.nu = res.x[2]

        # Calibration using SciPy's DifferentialEvolution
        elif calibration=="DE_SciPy":
            bnds = ( (0.001,2000) , (-0.999,0.999) , (0.001,2000)  )
            res = differential_evolution(objfunc, bounds = bnds,
                           args=(self.beta, self.F, self.K, self.time, self.vols),
                           maxiter=10000, popsize=40, mutation=(1.0, 1.5))

            self.alpha = res.x[0]
            self.rho = res.x[1]
            self.nu = res.x[2]

        # Calibration using the attached DifferentialEvolution algorithm - commented code available in ../src
        elif calibration == "DE_homebrew":
            bnds = ( (0.001,2000) , (-0.999,0.999) , (0.001,2000)  )
            res = list(DifferentialEvolution(objfunc, bounds = bnds,
                           args=(self.beta, self.F, self.K, self.time, self.vols),
                            mut=0.9, crossp=0.2, popsize=200, epochs=6000))

            self.alpha = res[-1][0][0]
            self.rho = res[-1][0][1]
            self.nu = res[-1][0][2]

            # we can also save the history of the loss function values.
            self.loss_history = [e[1] for e in res]

        else:
            raise Exception("Please insert a valid argument for `calibration` - ['SLS_SciPy', 'DE_SciPy', 'DE_homebrew']")

    def plot_smile(self):
        """
        Plot the volatility smile
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        # first, create a linspace on the SHIFTED strike prices
        x = np.linspace(np.min(self.K), np.max(self.K), num=100) + self.shift
        # then, estimate volatilities on that linspace
        y = [SABR(self.alpha, self.beta, self.rho, self.nu, self.F, k, self.time) for k in x]
        # lastly, un-shift the linspace
        x -= self.shift

        ax.set_title(f"Fitted volatility smile \n α={round(self.alpha, 2)}, β={round(self.beta, 2)}, ρ={round(self.rho, 2)}, ν={round(self.nu, 2)}")
        ax.set_ylabel("[Implied] Volatility")
        ax.set_xlabel("Strike Swaption Rate")
        ax.scatter(x = self.K, y = self.vols)
        # plot ATM rate in red
        ax.scatter(x = self.K[len(self.K)//2], y = self.vols[len(self.K)//2], c = "r")
        ax.plot(x, y)

    def predict(self):
        """
        Predicts volatility at the N strike prices, K.
        Takes the shift into account.
        """
        return [SABR(self.alpha, self.beta, self.rho, self.nu, self.F, k, self.time) for k in (self.K + self.shift)]
