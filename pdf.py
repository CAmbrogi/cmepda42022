import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):

    def __init__(self, x, y):
        '''
        Constructor
        '''
        super().__init__(x, y)
        interpolated = InterpolatedUnivariateSpline(x, y)

        intint = interpolated.integral(min(x), max(x))
        self.pdf = InterpolatedUnivariateSpline(x, y/intint)

        kvec = np.linspace(min(x), max(x), 100)

        self.cdf = np.array([self.pdf.integral(-np.inf, c) for c in kvec ])

        self.ppf = InterpolatedUnivariateSpline(self.cdf, kvec)


    def prob(self, uplim, lowlim):
        '''
        return the probability for the random variable to be included between
        up lim and low lim
        '''
        return self.pdf.integral(uplim, lowlim)

    def sample(self, lung):
        '''
        returns an array of random values from the ppf
        '''
        return self.ppf(np.random.rand(lung))

xs = np.linspace(-5, 5, 20)
ys = np.exp(-(xs**2)*0.5)

f = ProbabilityDensityFunction (xs, ys)

print(f.prob(0, 1))

plt.hist(f.sample(10000), bins='auto', density=True)

plt.plot(xs,ys, 'o')

plt.plot(xs, f.pdf(xs))
print(type(f.cdf))
plt.show()
