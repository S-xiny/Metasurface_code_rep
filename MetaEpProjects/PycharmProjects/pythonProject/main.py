# Create the data.

if __name__  == '__main__':
    def test_surf():
        """Test surf on regularly spaced co-ordinates like MayaVi."""
        def f(x, y):
            sin, cos = np.sin, np.cos
            return sin(x + y) + sin(2 * x - y) + cos(3 * x + 4 * y)

        x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
        s = surf(x, y, f)
        #cs = contour_surf(x, y, f, contour_z=0)
        return s


    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from mayavi.mlab import *
    from mayavi import mlab
    import numpy as np
    import os
    import re
    import glob
    import json
    import pickle
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import scipy
    from scipy.interpolate import griddata
    from scipy import interpolate
    from matplotlib import cm

    from matplotlib.pyplot import MultipleLocator

    #mlab.options.backend = 'envisage'
    #f = mlab.figure()
    # Define the functions for the two surfaces
    def f(x, y):
        return np.sin(np.sqrt(x**2 + y**2))

    def g(x, y):
        return np.cos(x + y)

    # Generate the data for the two surfaces
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z1 = f(X, Y)
    Z2 = g(X, Y)

    # Create the 3D plot

    # Plot the over part of surface 1 if it is over the surface 2
    s = mlab.mesh(X, Y, Z1*10, color = (0.9,0.1,0.1))

    # Plot the under part of surface 2 if it is under the surface 1
    s = mlab.mesh(X, Y, Z2, color = (0.1,0.1,0.8))
    mlab.show()