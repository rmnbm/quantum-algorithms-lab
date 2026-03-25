import matplotlib.pyplot as plt
import numpy as np

def plot_benchmark_comparison(df_results):
    
    labels = [f"Inst {i}" for i in df_results['Instance']]
    long_costs = df_results['Long_Run_Cost']
    short_costs = df_results['Best_of_Short_Runs']
    errors = df_results['Short_Runs_Std']

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.bar(x - width/2, long_costs, width, label='Strategy A: One Long Run', color='#2c3e50')
    
    ax.bar(x + width/2, short_costs, width, yerr=errors, label='Strategy B: Best of 10 Short Runs', 
           capsize=5, color='#e74c3c')

    ax.set_ylabel('Energy (Lower is better)')
    ax.set_title('Resource Efficiency Benchmark: Long Run vs. Short Runs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()