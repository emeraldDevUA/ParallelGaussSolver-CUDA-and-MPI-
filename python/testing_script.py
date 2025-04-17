import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (required for 3D plotting)
from tqdm import tqdm


def measure_execution_time(command_template, proc_counts, array_sizes, n_runs=3):
    results = {proc: [] for proc in proc_counts}
    errors = {}

    for proc in tqdm(proc_counts, desc="Processes"):
        error_messages = []

        for size in tqdm(array_sizes, desc=f"Array Sizes (proc={proc})", leave=False):
            total_time, valid_runs = 0.0, 0

            for _ in range(n_runs):
                command = command_template.format(proc=proc, size=size)
                start = time.time()
                process = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                end = time.time()

                if process.returncode == 0:
                    total_time += end - start
                    valid_runs += 1
                else:
                    error_messages.append(f"Error for {proc} processes, size {size}: {process.stderr.decode().strip()}")

            avg_time = total_time / valid_runs if valid_runs > 0 else float('inf')
            results[proc].append(avg_time)

        if error_messages:
            errors[proc] = error_messages

    if errors:
        print("Errors detected:")
        for proc, msgs in errors.items():
            for msg in msgs:
                print(msg)

    return results


def compute_speedup_and_efficiency(results_dict):
    base_times = np.array(results_dict[min(results_dict.keys())])
    speedup_dict = {}
    efficiency_dict = {}

    for proc, times in results_dict.items():
        times = np.array(times)
        speedup = base_times / times
        efficiency = speedup / proc
        speedup_dict[proc] = speedup.tolist()
        efficiency_dict[proc] = efficiency.tolist()

    return speedup_dict, efficiency_dict


def plot_line_chart(proc_counts, array_sizes, results, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for proc in proc_counts:
        plt.plot(array_sizes, results[proc], marker='o', label=f'{proc} Processes')

    plt.xlabel('Array Size')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_3d_surface(proc_counts, array_sizes, data_dict, zlabel, title, filename, cmap='viridis'):
    X, Y = np.meshgrid(proc_counts, array_sizes)
    Z = np.array([data_dict[proc] for proc in proc_counts]).T

    if Z.shape != X.shape:
        raise ValueError(f"Shape of data {Z.shape} does not match grid shape {X.shape}")

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')

    ax.set_xlabel('Number of Processes (p)')
    ax.set_ylabel('Array Size (n)')
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_speedup_efficiency(proc_counts, array_sizes, results):
    base_times = results[1]

    speedup = {
        proc: [base_times[i] / results[proc][i] for i in range(len(array_sizes))]
        for proc in proc_counts[1:]
    }

    efficiency = {
        proc: [s / proc for s in speedup[proc]]
        for proc in proc_counts[1:]
    }

    plot_line_chart(proc_counts[1:], array_sizes, speedup, 'Speedup (T1 / Tp)', 'MPI Speedup vs Array Size', '/app/output/speedup.png')
    plot_line_chart(proc_counts[1:], array_sizes, efficiency, 'Efficiency (Speedup / P)', 'MPI Efficiency vs Array Size', '/app/output/efficiency.png')


if __name__ == "__main__":
    command_template = "mpirun -np {proc} ./app/MPI/mpi_program {size}"
    proc_counts = [1, 2, 4, 6, 8]
    array_sizes = [1000 + 100 * i for i in range(1, 20)]

    results = measure_execution_time(command_template, proc_counts, array_sizes)
    plot_line_chart(proc_counts, array_sizes, results, 'Execution Time (s)', 'MPI Execution Time vs Array Size', '/app/output/execution_time.png')
    plot_3d_surface(proc_counts, array_sizes, results, 'Execution Time (s)', 'Execution Time vs Processes and Array Size', '/app/output/execution_time3d.png', cmap='inferno')

    speedup_dict, efficiency_dict = compute_speedup_and_efficiency(results)
    plot_3d_surface(proc_counts, array_sizes, speedup_dict, 'Speedup', 'Speedup vs Processes and Array Size', '/app/output/speedup3d.png', cmap='viridis')
    plot_3d_surface(proc_counts, array_sizes, efficiency_dict, 'Efficiency', 'Efficiency vs Processes and Array Size', '/app/output/efficiency3d.png', cmap='plasma')

    plot_speedup_efficiency(proc_counts, array_sizes, results)
