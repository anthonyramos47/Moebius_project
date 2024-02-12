import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import SmoothBivariateSpline

import warnings
warnings.simplefilter('ignore')

train_x, train_y = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5))
train_x = train_x.flatten()
train_y = train_y.flatten()

def z_func(x, y):
    return np.cos(x) + np.sin(y) ** 2 + 0.05 * x + 0.1 * y

train_z = z_func(train_x, train_y)
interp_func = SmoothBivariateSpline(train_x, train_y, train_z, s=0.0)
smth_func = SmoothBivariateSpline(train_x, train_y, train_z)

test_x = np.arange(-9, 9, 0.01)
test_y = np.arange(-9, 9, 0.01)
grid_x, grid_y = np.meshgrid(test_x, test_y)

interp_result = interp_func(test_x, test_y).T
smth_result = smth_func(test_x, test_y).T
perfect_result = z_func(grid_x, grid_y)

print(interp_result.shape)

fig, axes = plt.subplots(1, 3, figsize=(16, 8))
extent = [test_x[0], test_x[-1], test_y[0], test_y[-1]]
opts = dict(aspect='equal', cmap='nipy_spectral', extent=extent, vmin=-1.5, vmax=2.5)

im = axes[0].imshow(perfect_result, **opts)
fig.colorbar(im, ax=axes[0], orientation='horizontal')
axes[0].plot(train_x, train_y, 'w.')
axes[0].set_title('Perfect result, sampled function', fontsize=21)

im = axes[1].imshow(smth_result, **opts)
axes[1].plot(train_x, train_y, 'w.')
fig.colorbar(im, ax=axes[1], orientation='horizontal')
axes[1].set_title('s=default', fontsize=21)

im = axes[2].imshow(interp_result, **opts)
fig.colorbar(im, ax=axes[2], orientation='horizontal')
axes[2].plot(train_x, train_y, 'w.')
axes[2].set_title('s=0', fontsize=21)


plt.show()