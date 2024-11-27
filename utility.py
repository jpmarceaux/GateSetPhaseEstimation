import numpy as np

def angular_difference(phi1, phi2, domain=2*np.pi):
    # returns the distance between two angles measured in radians 
    diff = phi2 - phi1
    return np.mod(diff + domain/2, domain) - domain/2


def make_fibonacci_sequence(n):
    a = 1
    b = 1
    sequence = [a,b]
    for i in range(n-2):
        c = a+b
        sequence.append(c)
        a = b
        b = c
    return sequence[1:]