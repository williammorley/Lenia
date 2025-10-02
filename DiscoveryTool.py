#Discovery Tool for Asymptotic Lenia

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#--------------------------------------------------
#Implementation identical to other Asymptotic Lenia
#--------------------------------------------------
size = 64
T = 10
R = 13
steps = 100
GS = 13
#Final grid mean value cut offs
MEAN_BOUNDS = (0.01, 0.1)
NUM_SEEDS = 500

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

def update_grid(grid, m, s):
    #FFT Convolution update - Euler step.
    U = np.real(np.fft.ifft2(fK * np.fft.fft2(grid)))
    return grid + (1 / T) * (target(U, m, s) - grid)


#----------------------------------------------------------------
#New code
#----------------------------------------------------------------
def discover_seeds():
    #User inputs m and s range
    m_min = float(input("Enter minimum m (e.g., 0.12): "))
    m_max = float(input("Enter maximum m (e.g., 0.16): "))
    s_min = float(input("Enter minimum s (e.g., 0.012): "))
    s_max = float(input("Enter maximum s (e.g., 0.016): "))

    #empty array to save good seeds
    accepted_data = []

    for seed_num in range(NUM_SEEDS):
        grid = np.zeros((size, size))
        #create a subarray that is filled with random numbers 0-1
        patch = np.random.rand(GS, GS)
        #find centre
        x0 = (size - GS) // 2
        #fill centre with subarray 'patch'
        grid[x0:x0+GS, x0:x0+GS] = patch
        init_grid = grid.copy()

        #select random m,s values from array.
        m = np.random.uniform(m_min, m_max)
        s = np.random.uniform(s_min, s_max)

        for _ in range(steps):
            grid = update_grid(grid, m, s)

        mean_val = np.mean(grid)
        #write data if mean is in accepted range
        if MEAN_BOUNDS[0] <= mean_val <= MEAN_BOUNDS[1]:
            accepted_data.append({
                'init': init_grid,
                'final': grid,
                'm': m,
                's': s
            })
    #save file with name of user's choice.
    save_name = input("Enter filename to save (e.g. seeds.npz): ")
    np.savez(save_name, **{
        f"sample_{i}": accepted_data[i] for i in range(len(accepted_data))
    })
    print(f"Saved {len(accepted_data)} seeds to {save_name}")

    display_results(accepted_data)

def display_results(data_list):
    #Limit to the first 16 items for disply in final report
    data_list = data_list[:16]  
    n = len(data_list)
    cols = 4 
    rows = (n + cols - 1) // cols

    plt.figure(figsize=(3*cols, 3*rows))
    for i, d in enumerate(data_list):
        plt.subplot(rows, cols, i+1)
        plt.imshow(d['final'], cmap='viridis')
        #pad 2 moves title closer to figure than normal
        plt.title(f"{i}: m={d['m']:.3f}\ns={d['s']:.4f}", fontsize=6, pad=2)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('seeds_16.png', dpi=300)
    plt.show()


def load_and_animate():
    fname = input("Enter .npz filename: ")
    data = np.load(fname, allow_pickle=True)
    #pull sample from npz using user input key
    samples = [data[key].item() for key in data]

    #give user choce, display figure of all samples or test sample direct
    print("\nChoose how to proceed:")
    print("1. Enter sample number directly")
    print("2. Display all samples before choosing")

    mode = input("Enter 1 or 2: ")

    if mode == "2":
        display_results(samples)

    try:
        sample_num = int(input("Enter the number of the sample to animate: "))
        selected = samples[sample_num]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    grid = selected['init']
    m = selected['m']
    s = selected['s']

    #animate with
    print(f"\nAnimating sample {sample_num} with m={m:.4f}, s={s:.4f}")

    fig, ax = plt.subplots()
    im = ax.imshow(grid, cmap='viridis', animated=True)
    ax.axis('off')

    def update(frame):
        #produce animation for selected seed
        nonlocal grid
        grid = update_grid(grid, m, s)
        im.set_array(grid)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    ani.save("seed10.gif", writer='pillow', fps=30)
    plt.show()

    #After animation, extract 13x13 seed
    x0 = (size - GS) // 2
    seed_patch = selected['init'][x0:x0+GS, x0:x0+GS]

    print("\n--- Sample Info ---")
    print(f"m = {m:.4f}")
    print(f"s = {s:.4f}")
    print("13x13 Starting Seed:")
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print(seed_patch)

    #Give user choice to save file
    save_choice = input("\nSave this info to a text file? (y/n): ").lower()
    if save_choice == 'y':
        out_fname = input("Enter filename (e.g., seed_info.txt): ")
        with open(out_fname, 'w') as f:
            f.write(f"Sample {sample_num}\n")
            f.write(f"m = {m:.4f}\n")
            f.write(f"s = {s:.4f}\n")
            f.write("13x13 Starting Seed:\n")
            np.savetxt(f, seed_patch, fmt="%.3f")
        print(f"Saved to {out_fname}")



def main():
    print("1. Discover new seeds")
    print("2. Load existing .npz and animate a seed")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        discover_seeds()
    elif choice == "2":
        load_and_animate()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()

