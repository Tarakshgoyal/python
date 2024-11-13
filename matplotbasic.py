import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
y=[1,8,27,4*4*4,5*5*5,6*6*6,7*7*7,8*8*8,9*9*9,1000]
x=[1,2,3,4,5,6,7,8,9,10]
plt.plot(x,y, label='x^3', color='red', linewidth=2,marker='.',linestyle='--',markersize=10,markeredgecolor='blue')
x2=np.arange(0,11,1)
print(x2)
plt.plot(x2,2*x2**3,label='2x^3',color='blue',linewidth=2,marker='.',markersize=10,markeredgecolor='red')
plt.title('graph of x^3 ' )
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.xticks([0,1,2,3,4,5,6,7,8,9,10])
plt.yticks([1,8,27,4*4*4,5*5*5,6*6*6,7*7*7,8*8*8,9*9*9,1000])
plt.legend()
plt.show()