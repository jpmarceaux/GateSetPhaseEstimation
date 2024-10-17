import numpy as np

def angular_difference(phi1, phi2):
    # returns the distance between two angles measured in radians 
    diff = phi2 - phi1
    return np.mod(diff + np.pi, 2*np.pi) - np.pi