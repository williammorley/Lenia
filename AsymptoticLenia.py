#Implementation of Asymptotic Lenia using FFT to improve efficiency.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Parameter Presets
size = 64
T = 10
steps = 200  
m, s = 0.2, 0.02
R = 21

def gaussian(x, m, s): 
    #defines Gaussian for use as target function
    return np.exp(-((x - m) / s) ** 2 / 2)

#Create circular kernel, apply Bell curve to kernel radially, normalise the kernel
y, x = np.ogrid[-R:R+1, -R:R+1]
D = np.sqrt(x**2 + y**2) / R
K = (D < 1) * gaussian(D, 0.5, 0.15)
K = K / np.sum(K)

#Pad kernel to grid size, compute FFT of kernel
K_pad = np.zeros((size, size))
k_size = K.shape[0]
K_pad[:k_size, :k_size] = K
K_pad = np.roll(K_pad, -k_size//2, axis=(0, 1))
fK = np.fft.fft2(K_pad)

def target(U, m, s):
    #Target growth profile - defined for modularity
    return gaussian(U, m, s)

def solver_explicit(grid, T, steps):
    #Standard Euler step
    results = []
    for _ in range(steps):
        U = np.real(np.fft.ifft2(fK * np.fft.fft2(grid)))
        grid += (1 / T) * (target(U, m, s) - grid)
        results.append(np.copy(grid))
    return results

def solver_rk2(grid, T, steps):
    #Runge-Kutta 2 method
    results = []
    for _ in range(steps):
        U1 = np.real(np.fft.ifft2(fK * np.fft.fft2(grid)))
        k1 = (target(U1, m, s) - grid)
        mid = grid + 0.5 * (1 / T) * k1
        U2 = np.real(np.fft.ifft2(fK * np.fft.fft2(mid)))
        k2 = (target(U2, m, s) - mid)
        grid += (1 / T) * k2
        results.append(np.copy(grid))
    return results

def solver_rk4(grid, T, steps):
    #Runge-Kutta 4 method
    results = []
    def get_rhs(state):
        U = np.real(np.fft.ifft2(fK * np.fft.fft2(state)))
        return target(U, m, s) - state
    for _ in range(steps):
        k1 = get_rhs(grid)
        k2 = get_rhs(grid + 0.5 * (1 / T) * k1)
        k3 = get_rhs(grid + 0.5 * (1 / T) * k2)
        k4 = get_rhs(grid + (1 / T) * k3)
        grid += (1 / T) * (k1 + 2*k2 + 2*k3 + k4) / 6
        results.append(np.copy(grid))
    return results


#Method choice using menu. Key assigned to methods using a dict.
methods = {
    "1": ("Explicit Euler", solver_explicit),
    "2": ("Runge-Kutta 2 (RK2)", solver_rk2),
    "3": ("Runge-Kutta 4 (RK4)", solver_rk4)
}

print("Choose a solver method:")
for key, (name, _) in methods.items():
    print(f"{key}: {name}")
choice = input("Enter the number of the method you want to use: ").strip()

if choice not in methods:
    raise ValueError("Invalid selection. Please run the script again.")

method_name, solver = methods[choice]

#Run selected method
initial_grid = np.random.rand(size, size)
results = solver(np.copy(initial_grid), T, steps)

#Set up plot for animation
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(results[0], cmap='plasma')
ax.set_title(f'{method_name} - Frame 1')
ax.axis('off')

def update(frame):
    #Include frame name in animation title
    img.set_array(results[frame])
    ax.set_title(f'{method_name} - Frame {frame+1}')

ani = animation.FuncAnimation(fig, update, frames=steps, interval=30)

#Save as GIF
gif_filename = f"{method_name}.gif"
ani.save(gif_filename, writer='pillow', fps=30)
print(f"Saved animation as {gif_filename}")
