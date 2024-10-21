import numpy as np

def angular_difference(phi1, phi2, domain=2*np.pi):
    # returns the distance between two angles measured in radians 
    diff = phi2 - phi1
    return np.mod(diff + domain/2, domain) - domain/2