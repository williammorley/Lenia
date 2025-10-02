#Implementation of Lenia
#Core functions - update, growth, kernel etc. based on code written by Bert Chan in his jupyter notebook tutorial
#Uses Fast fourier transforms instead of direct convolution

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import zoom


""" Loading presets.
     Values for orbium taken from the Lenia Taxonomy.
      Random grid generates numbers between 0 and 1."""

def load_orbium():
    """Returns the Orbium grid configuration."""
    orbium = {
        "R": 13,
        "T": 10,
        "m": 0.120,
        "s": 0.0120,  
        "b": [1],
        "cells": np.array([
            [0,0,0,0,0,0,0.1,0.14,0.1,0,0,0.03,0.03,0,0,0.3,0,0,0,0],
            [0,0,0,0,0,0.08,0.24,0.3,0.3,0.18,0.14,0.15,0.16,0.15,0.09,0.2,0,0,0,0],
            [0,0,0,0,0,0.15,0.34,0.44,0.46,0.38,0.18,0.14,0.11,0.13,0.19,0.18,0.45,0,0,0],
            [0,0,0,0,0.06,0.13,0.39,0.5,0.5,0.37,0.06,0,0,0,0.02,0.16,0.68,0,0,0],
            [0,0,0,0.11,0.17,0.17,0.33,0.4,0.38,0.28,0.14,0,0,0,0,0,0.18,0.42,0,0],
            [0,0,0.09,0.18,0.13,0.06,0.08,0.26,0.32,0.32,0.27,0,0,0,0,0,0,0.82,0,0],
            [0.27,0,0.16,0.12,0,0,0,0.25,0.38,0.44,0.45,0.34,0,0,0,0,0,0.22,0.17,0],
            [0,0.07,0.2,0.02,0,0,0,0.31,0.48,0.57,0.6,0.57,0,0,0,0,0,0,0.49,0],
            [0,0.59,0.19,0,0,0,0,0.2,0.57,0.69,0.76,0.76,0.49,0,0,0,0,0,0.36,0],
            [0,0.58,0.19,0,0,0,0,0,0.67,0.83,0.9,0.92,0.87,0.12,0,0,0,0,0.22,0.07],
            [0,0,0.46,0,0,0,0,0,0.7,0.93,1,1,1,0.61,0,0,0,0,0.18,0.11],
            [0,0,0.82,0,0,0,0,0,0.47,1,1,0.98,1,0.96,0.27,0,0,0,0.19,0.1],
            [0,0,0.46,0,0,0,0,0,0.25,1,1,0.84,0.92,0.97,0.54,0.14,0.04,0.1,0.21,0.05],
            [0,0,0,0.4,0,0,0,0,0.09,0.8,1,0.82,0.8,0.85,0.63,0.31,0.18,0.19,0.2,0.01],
            [0,0,0,0.36,0.1,0,0,0,0.05,0.54,0.86,0.79,0.74,0.72,0.6,0.39,0.28,0.24,0.13,0],
            [0,0,0,0.01,0.3,0.07,0,0,0.08,0.36,0.64,0.7,0.64,0.6,0.51,0.39,0.29,0.19,0.04,0],
            [0,0,0,0,0.1,0.24,0.14,0.1,0.15,0.29,0.45,0.53,0.52,0.46,0.4,0.31,0.21,0.08,0,0],
            [0,0,0,0,0,0.08,0.21,0.21,0.22,0.29,0.36,0.39,0.37,0.33,0.26,0.18,0.09,0,0,0],
            [0,0,0,0,0,0,0.03,0.13,0.19,0.22,0.24,0.24,0.23,0.18,0.13,0.05,0,0,0,0],
            [0,0,0,0,0,0,0,0,0.02,0.06,0.08,0.09,0.07,0.05,0.01,0,0,0,0,0]
        ])
    }
    GS = 64
    R, T, m, s = orbium['R'], orbium['T'], orbium['m'], orbium['s']


    #Lenia is scale invariant, so can change magnification of monsters
    C = zoom(orbium['cells'], 1, order=0)


    #Centre the Orbium patten on an otherwise empty grid.
    A = np.zeros((GS, GS))
    cx, cy = (GS - C.shape[0]) // 2, (GS - C.shape[1]) // 2
    A[cx:cx+C.shape[0], cy:cy+C.shape[1]] = C

    return A, R, T, m, s

def load_random():
    """Generates a GS x GS grid of random numbers between 0 and 1."""
    GS = 64
    return np.random.rand(GS, GS)


def gaussian(x, m, s):
    """Definition of Gaussian for use in growth function and kernel"""
    return np.exp(-((x - m) / s) ** 2 / 2)


def growth(U, m, s, T):
    """Growth function as defined in Bert Chan's original Lenia Paper"""
    return gaussian(U, m, s) * 2 - 1


def build_k(R):
    """Create circular kernel
       Apply gaussian curve to kernel radially
       Normalise the kernel"""
    Y, X = np.ogrid[-R:R, -R:R]
    D = np.sqrt(X**2 + Y**2) / R
    K = (D < 1) * gaussian(D, 0.5, 0.15)
    K /= np.sum(K)
    return K


def pad_kernel(K, GS):
    """Pads and centres the kernel K into a GS x GS array ready for FFT convolution."""
    K_pad = np.zeros((GS, GS))
    K_shape = K.shape
    K_pad[:K_shape[0], :K_shape[1]] = K
    K_pad = np.roll(K_pad, -K_shape[0]//2, axis=0)
    K_pad = np.roll(K_pad, -K_shape[1]//2, axis=1)
    return np.fft.fft2(K_pad)


def ugrid(grid, fK, T, m, s):
    """Updates grid using multiplication of fourier transformed grid and kernel
        Updates grid using Euler step, 1/T = dt
        Clips values between 0 and 1 to restrict growth"""
    U = np.real(np.fft.ifft2(fK * np.fft.fft2(grid)))
    return np.clip(grid + (1 / T) * growth(U, m, s, T), 0, 1)


def generate_frames(A, R, T, m, s, GS):
    """Pre-compute all grid frames for the animation"""
    frames = [A]
    fK = pad_kernel(build_k(R), GS)

    for _ in range(150):
        frames.append(ugrid(frames[-1], fK, T, m, s))

    return frames


def animate(i, frames, im, text):
    """Take images of all grids and label them"""
    im.set_data(frames[i])

    #label frames with their frame number
    text.set_text(f'Frame: {i}')
    return [im, text]


def save_anim(frames, grid_type):
    """Assemble and save animation with filename based on grid type."""
    filename = f"{grid_type}Close.gif"
    fig, ax = plt.subplots()
    #viridis, plasma
    im = ax.imshow(frames[0], cmap='plasma')
    text = ax.text(2, 2, 'Frame: 0', color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.6))
    ax.axis('off')

    #Create animation with 50ms interval between frames.
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50, fargs=(frames, im, text), blit=True)
    plt.show()

    #Save file.
    ani.save(filename, writer='pillow', fps=20)


def generate_mean(A, R, T, m, s, GS):
    """Computes frames and calculates the mean value of the final grid cells"""
    frames = [A]
    fK = pad_kernel(build_k(R), GS)

    for _ in range(100):
        frames.append(ugrid(frames[-1], fK, T, m, s))

    #save data of final grid ready for averaging
    final_grid = frames[-1]

    return np.clip(np.mean(final_grid), 0, 1)


def plot_contour(A, R, T, GS):
    """Contour plot that sweeps through m and s values, plots mean cell value after 100 iterations
        on the Z axis (colour)."""
    m_values = np.arange(0.14, 0.16, 0.001)
    s_values = np.arange(0.014, 0.016, 0.0001)
    M, S = np.meshgrid(m_values, s_values)
    Z = np.zeros_like(M)

    
    for i in range(len(m_values)):
        for j in range(len(s_values)):
            m = m_values[i]
            s = s_values[j]
            mean_value = generate_mean(A, R, T, m, s, GS)
            Z[j, i] = mean_value

    #Figure Settings. vmin/vmax sets the scale of the colours.
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(M, S, Z, cmap='plasma') #, vmin=0, vmax=1)
    cbar = plt.colorbar(cp)
    cbar.set_label('Mean Grid Value')
    plt.title('Contour Plot of Mean Grid Value After 100 Iterations')
    plt.xlabel('m Value')
    plt.ylabel('s Value')
    plt.savefig('NewRandomContour.png', dpi=300)

    #Save data to npz file.
    np.savez('UltraFineOrbiumContour_data.npz', M=M, S=S, Z=Z)

    plt.show()

    

def main():
    """Simple menu to allow user to choose starting conditions. Could easily be extended to include more options"""
    while True:
        print("Choose grid type:")
        print("1. Orbium Grid (Animation)")
        print("2. Random Grid (Animation)")
        print("3. Contour Simulation (Plot)")
        choice = input("Enter 1, 2, or 3: ")

        if choice == '1':
            A, R, T, m, s = load_orbium()
            gtype = "orbium"
            GS = 64
            frames = generate_frames(A, R, T, m, s, GS)
            save_anim(frames, gtype)
            break
        elif choice == '2':
            A = load_random()
            m_val = float(input("Enter an m value to be tested:"))
            s_val = float(input("Enter an s value to be tested:"))
            #Parameters for random grid - R, T chosen to be the Orbium values, R, T, m, s = 13, 10, 0.15, 0.015
            R, T, m, s = 13, 10, m_val, s_val                                                                                                                
            gtype = "random"  
            GS = 64
            frames = generate_frames(A, R, T, m, s, GS)
            save_anim(frames, gtype)
            break
        elif choice == '3':
            #A = load_random()
            A, *_ = load_orbium()  #can be orbium/random to check stability
            R, T= 13, 10           #Parameters for random grid - chosen to be the Orbium values, 
            GS = 64
            plot_contour(A, R, T, GS)
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == '__main__':
    main()
