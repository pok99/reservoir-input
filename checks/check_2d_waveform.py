import numpy as np

from mpl_toolkits import mplot3d 
import matplotlib.pyplot as plt

import pdb



x = np.outer(np.linspace(-6, 6, 1000), np.ones(1000)) 
y = x.copy().T # transpose 
z = .1 * x * y * (np.sin(x * y)) 

  
# # Creating figyre 
# fig = plt.figure(figsize =(14, 9)) 
# ax = plt.axes(projection ='3d') 
  
# # Creating plot 
# ax.plot_surface(x, y, z) 
# ax.set_zlim3d(-5,5)
  
# # show plot 
# plt.show() 


plt.imshow(z, cmap='hot', interpolation='nearest')
plt.show()