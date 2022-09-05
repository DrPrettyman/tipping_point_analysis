"""
The module ``noise_methods`` provides various functions for
generating stochastic processes or noise signals.

Includes:
 - White noise
 - Random walk
 - The AR(1) model (autoregressive model with lag 1)
   commonly used to model a system undergoing Critical Slowing Down
 - The AR(63) model
   Used to model noise of varying colour
"""
#
#
#
import matplotlib.pyplot as plt
import numpy as np
from tippingpoints import scaling_methods


def white_noise(n: int, eta: float = 1):
    """
    Returns a simple gaussian white noise signal where each term is
    sampled independently from a normal distribution

    :param n:   length of returned series
    :param eta: standard deviation of series
    :returns: ``numpy.ndarray``
    """
    return eta * np.random.randn(n)


def random_walk(n: int, eta=1):
    """
    Returns a simple random walk, the cumulative sum of a gaussian white noise series

    :param n:   length of returned series
    :param eta: standard deviation of the white noise series
    :returns: ``numpy.ndarray``
    """
    return np.cumsum(eta * np.random.randn(n))


def ar1(n: int, mu: float = 1., eta: float = 1.):
    """
    Returns an AR(1) signal of length ``n`` with autoregressive parameter ``mu``

    :param n:   Length og the series
    :param mu:  Autoregressive parameter :math:`\\mu`. Defaults to 1.0 giving a random walk
    :param eta: Standard deviation of noise process, defaults to 1.

    :returns: AR(1) signal of length ``n``
    """
    z = np.zeros(n)
    for i in range(1, n):
        z[i] = mu * z[i - 1] + eta * np.random.randn()
    return z


def ar63(n: int, lamb: float = 2., eta: float = 1.):
    """
    Returns as AR(63) series  

    :param n:
    :param lamb:
    :param eta:
    :return:
    """
    a = np.zeros(64)
    a[0] = 1.
    for k in range(1, 64):
        a[k] = (k - 1 - lamb / 2) * a[k - 1] / k
    z = np.zeros(n + 164)
    for t in range(64, n + 164):
        z[t] = eta * np.random.randn() - \
               (np.flip(a[1::]) * z[t - 63:t]).sum()
    return z[164::]


# Tests:


def ar63_test():
    print('')
    print('AR(63) test:')
    n = 10 ** 5
    t = np.linspace(0, 1, n)
    print('')
    print('Simple random walk:')
    z = random_walk(n)
    pse_value, f, ps = scaling_methods.pse(z, binning=True)
    print('pse value = {}'.format(pse_value))
    fig0, ax = plt.subplots(2)
    ax[0].plot(t, z, )
    ax[1].plot(f, ps, )
    ax[0].set(xlabel='t', ylabel='z', title='Simple random walk')
    ax[1].set(xlabel='log10(f)', ylabel='log10(psdx)')
    plt.show()
    print('')
    print('Three examples,')
    test_lambda_values = [0, 1, 2]
    print('using test lambda values {}, {}, {}'
          .format(test_lambda_values[0], test_lambda_values[1], test_lambda_values[2]))
    z0 = ar63(n, lamb=test_lambda_values[0])
    z1 = ar63(n, lamb=test_lambda_values[1])
    z2 = ar63(n, lamb=test_lambda_values[2])
    fig1, ax = plt.subplots(3)
    ax[0].plot(t, z0, )
    ax[1].plot(t, z1, )
    ax[2].plot(t, z2, )
    ax[2].set(xlabel='time (t)', ylabel='z')
    plt.show()
    pse0, f0, ps0 = scaling_methods.pse(z0)
    pse1, f1, ps1 = scaling_methods.pse(z1)
    pse2, f2, ps2 = scaling_methods.pse(z2)
    print('pse values: {0}, {1}, {2}'.format(pse0, pse1, pse2))
    fig2, ax = plt.subplots(3)
    ax[0].plot(f0, ps0, )
    ax[1].plot(f1, ps1, )
    ax[2].plot(f2, ps2, )
    ax[2].set(xlabel='time (t)', ylabel='z')
    plt.show()
    print('')
    print('Test:')
    lambda_values = np.linspace(0, 2, 100)
    pse_values = np.zeros(100)
    for i in range(100):
        if i % 10 == 0:
            print(i)
        z = ar63(10 ** 4, lamb=lambda_values[i])
        pse_value, f, ps = scaling_methods.pse(z, binning=True)
        pse_values[i] = pse_value
    print('We expect a one-to-one relationship between the test lambda values\n'
          'and the pse values returned by the pse() method')
    fig3, ax = plt.subplots()
    ax.plot(lambda_values, pse_values, )
    ax.set(xlabel='lambda', ylabel='pse value')
    plt.show()
    return
