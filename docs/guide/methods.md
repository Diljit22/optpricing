# Methods

One formula for each numerical method the library actually runs, and the paper
it is taken from. The code is the reference for signs and special cases; this
page is the map.

## Black–Scholes–Merton

With continuous dividend yield \(q\),

\[
C = Se^{-qT}\Phi(d_1) - Ke^{-rT}\Phi(d_2),
\quad
d_{1,2} = \frac{\ln(S/K)+(r-q\pm\sigma^2/2)T}{\sigma\sqrt{T}}.
\]

Black and Scholes, *Journal of Political Economy* (1973); Merton, *Bell Journal
of Economics and Management Science* (1973). Puts follow by put-call parity.

## Heston characteristic function

The variance factor of \(\varphi(u) = \mathbb{E}[e^{iu\ln S_T}]\) is
\(C(u)+D(u)v_0\), in the "little trap" branch (the root is taken so that the
ratio \(g\) stays inside the unit disk):

\[
\beta = \kappa - \rho\sigma iu, \quad
d = \sqrt{\beta^2 + \sigma^2(iu+u^2)}, \quad
g = \frac{\beta-d}{\beta+d}.
\]

\[
C = \frac{\kappa\theta}{\sigma^2}\Big[(\beta-d)T - 2\ln\frac{1-ge^{-dT}}{1-g}\Big],
\quad
D = \frac{\beta-d}{\sigma^2}\frac{1-e^{-dT}}{1-ge^{-dT}}.
\]

At \(u\in\{0,-i\}\) the forcing term vanishes and \(C=D=0\). That point is
\(0/0\) in the original Heston formula when \(\rho\sigma\ge\kappa\).

Heston, *Review of Financial Studies* (1993). Albrecher, Mayer, Schoutens and
Tistaert, "The little Heston trap", *Wilmott* (2007).

## COS

Fang and Oosterlee expand the density of \(y=\ln S_T\) in cosines on a finite
interval \([a,b]\). With \(u_k = k\pi/(b-a)\),

\[
e^{rT}V(x) \approx \sum_{k=0}^{N-1}{}' \operatorname{Re}\big[\varphi(u_k)e^{-iu_k a}\big]\,V_k,
\]

where the prime halves the \(k=0\) term. The put coefficients \(V_k\) are
closed form (\(\chi\) and \(\psi\) of the payoff \((K-e^y)^+\)). Calls come from
put-call parity, because the call payoff is not bounded on the truncation
interval.

The interval is the Chernoff bound on the tails from the moment strip of
\(\varphi\), and falls back to the cumulant rule \(c_1 \pm L\sqrt{c_2+\sqrt{c_4}}\)
when a side has no usable moments. \(N\) grows until the tail of \(\varphi\) is
small, and stops at \(2^{16}\) with a warning.

Fang and Oosterlee, *SIAM Journal on Scientific Computing* 31 (2008).

## Carr–Madan FFT

The Fourier transform of the damped call (or put, or covered call) is

\[
\psi(v) = \frac{e^{-rT}\varphi\big(v-(\alpha+1)i\big)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}.
\]

\(\alpha\) is not fixed at \(1.75\). It minimises the Lord–Kahl objective
\(-\alpha k + \ln\mathbb{E}[S_T^{\alpha+1}] - \ln|\alpha(\alpha+1)|\) inside the
strip where that moment exists: \(\alpha>0\) for a call, \(\alpha<-1\) for a
put, and \(-1<\alpha<0\) for \(C - Se^{-qT}\). The grid is centred on the
strike and extended until \(\varphi\) has decayed.

Carr and Madan, *Journal of Computational Finance* (1999). Lord and Kahl,
"Optimal Fourier inversion in semi-analytical option pricing" (2007). Lee,
*Journal of Computational Finance* (2004), for which payoff the damping prices.

## Crank–Nicolson with Rannacher start-up

On the Black–Scholes operator \(A\), one step of size \(\Delta t\) is

\[
(I - \theta\Delta t\, A)V^{n+1} = (I + (1-\theta)\Delta t\, A)V^{n},
\quad \theta = \tfrac12.
\]

The payoff kink makes plain Crank–Nicolson oscillate, and gamma loses an
order. The first two steps are each replaced by two fully implicit half-steps
(\(\theta=1\), width \(\Delta t/2\)). The spot is placed on a grid node, so the
one-sided differences for delta and gamma are taken at \(S_0\).

Crank and Nicolson, *Proceedings of the Cambridge Philosophical Society*
(1947). Rannacher, *Computing* (1984).

## Leisen–Reimer

The up and down probabilities are the Peizer–Pratt inversion of the
Black–Scholes \(d_1\) and \(d_2\), which cancels the \(O(1/N)\) error of CRR at
any strike. For odd \(N\),

\[
h^{-1}(z) = \tfrac12 + \operatorname{sign}(z)\,\tfrac12
\sqrt{1-\exp\!\Big[-\Big(\frac{z}{N+\tfrac13+\frac{0.1}{N+1}}\Big)^2
\big(N+\tfrac16\big)\Big]}.
\]

\(p = h^{-1}(d_2)\) is the risk-neutral probability and \(h^{-1}(d_1)\) sets the
spacing. An even request is run as the next odd count.

Leisen and Reimer, *Applied Mathematical Finance* (1996). Peizer and Pratt,
*Journal of the American Statistical Association* (1968).

## Longstaff–Schwartz

Paths are simulated on a time grid. Working backward, the continuation value on
the in-the-money paths is the least-squares fit of the discounted future cash
flow on the monomials \(1, x, x^2, \ldots\) in the centred moneyness
\(x = S/K - 1\) (the same polynomials as powers of \(S/K\)). Exercise when the
intrinsic value is larger than that fit.

The regression is summed in a fixed number of chunks and reduced in chunk
order, so the exercise decision does not depend on the thread count.

Longstaff and Schwartz, *Review of Financial Studies* (2001).

## Andersen quadratic-exponential

The Heston variance over a step has mean \(m\) and variance \(s^2\). With
\(\psi = s^2/m^2\) and a threshold \(\psi_c = 1.5\),

\[
\psi \le \psi_c:\quad V = a\big(\sqrt{b}+Z\big)^2,
\qquad
\psi > \psi_c:\quad
V = \begin{cases} 0 & U \le p, \\ \ln\frac{1-p}{1-U}/\beta & U > p. \end{cases}
\]

\(Z\) is standard normal and \(U\) is uniform. The log-spot increment uses
Andersen's martingale correction, so the simulated asset matches the discount
curve when the correction is defined. This is the default
`variance_scheme="qe"`. `"euler"` is full-truncation Euler: the variance may
go negative, and only the positive part enters the drift and the diffusion.

Andersen, "Simple and efficient simulation of the Heston stochastic volatility
model", *Journal of Computational Finance* (2008). Lord, Koekkoek and van Dijk,
*Applied Mathematical Finance* (2010), for the Euler scheme.
