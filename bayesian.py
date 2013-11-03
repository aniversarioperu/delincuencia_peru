# -*- coding: utf-8 -*-
import prettyplotlib as ppl
from prettyplotlib import plt
import sys
import pymc as pm
import numpy as np

datos = np.loadtxt("datos.txt")
alpha = 1.0/datos.mean()
print alpha
print "alpha %f" % alpha
print "datos.mean %f" % datos.mean()

n_datos = len(datos)
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)
print lambda_1.random()
print lambda_2.random()

tau = pm.DiscreteUniform("tau", lower=0, upper=n_datos)
print tau.random()

@pm.deterministic
def lambda_(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
    out = np.zeros(n_datos)
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

observation = pm.Poisson("obs", lambda_, value=datos, observed=True)
model = pm.Model([observation, lambda_1, lambda_2, tau])
mcmc = pm.MCMC(model)
mcmc.sample(50000, 10000, 1)

lambda_1_samples = mcmc.trace('lambda_1')[:]
lambda_2_samples = mcmc.trace('lambda_2')[:]

tau_samples = mcmc.trace('tau')[:]

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)

plt.rc('font', **{'family': 'DejaVu Sans'})
plt.subplot(311)
plt.title(u'''Distribución posterior de las variables
            $\lambda_1,\;\lambda_2,\;tau$''')
plt.hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=0.85,
        normed=True)
plt.xlim([150000,250000])
plt.xlabel("valor de $\lambda_1$")


plt.subplot(312)
#ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=0.85,
        normed=True)
plt.xlim([150000,250000])
plt.xlabel("valor de $\lambda_2$")
plt.tick_params(axis="both", which="mayor", labelsize=4)

plt.subplot(313)
w = 1.0/tau_samples.shape[0]*np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_datos, alpha=1, weights=w, rwidth=2.0)
plt.xticks(np.arange(n_datos))
plt.ylim([0, 1.5])
plt.xlim([0, 8])
plt.xlabel("valor de $tau$")

fig.set_size_inches(7,6)
fig.tight_layout()
fig.savefig("plot1.png")



fig, ax = plt.subplots(nrows=1, ncols=1)
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_datos)
for day in range(0, n_datos):
    ix = day < tau_samples
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N

anhos = ["2005","2006","2007","2008","2009","2010","2011","2012"]

plt.plot(range(n_datos), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_datos)
plt.xticks(np.arange(n_datos) + 0.4, anhos)
plt.xlabel(u'Años')
plt.ylabel(u'Número esperado de delitos')
plt.title(u'''Cambio en el número esperado de delitos por año''')
plt.ylim(0, 300000)
plt.bar(np.arange(len(datos)), datos, color="#348ABD", alpha=0.65)

#plt.legend(loc="upper left")
fig.savefig("plot2.png")
