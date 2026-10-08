import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    perch_path = os.path.join(base_dir, 'combined_results_perch.csv')
    aves_path = os.path.join(base_dir, 'combined_results_aves.csv')

    # Load data
    df_perch = pd.read_csv(perch_path)
    df_aves = pd.read_csv(aves_path)

    # Cap at 2 seconds
    df_perch = df_perch[df_perch['window_sec'] <= 2.0]
    df_aves = df_aves[df_aves['window_sec'] <= 2.0]

    # Group by window_sec and get max nmi
    perch_max_nmi = df_perch.groupby('window_sec')['nmi'].max().reset_index()
    aves_max_nmi = df_aves.groupby('window_sec')['nmi'].max().reset_index()

    # Sort by window_sec to ensure the line plot is ordered
    perch_max_nmi = perch_max_nmi.sort_values('window_sec')
    aves_max_nmi = aves_max_nmi.sort_values('window_sec')

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(perch_max_nmi['window_sec'], perch_max_nmi['nmi'], marker='o', label='Perch', linewidth=2)
    plt.plot(aves_max_nmi['window_sec'], aves_max_nmi['nmi'], marker='s', label='Aves', linewidth=2)

    plt.xlabel('Window Size (seconds)', fontsize=24)
    plt.ylabel('Max NMI', fontsize=24)
    plt.title('Max NMI vs Window Size', fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save image
    output_path = os.path.join(base_dir, 'nmi_comparison.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    main()
