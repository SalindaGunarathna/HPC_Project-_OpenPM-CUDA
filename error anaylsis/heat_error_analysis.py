import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

def load_heat_data(filename):

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found!")
        return None, None

    metadata = {}

    # Read metadata from header comments
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#'):
                if 'Grid Size:' in line:
                    size_str = line.split(':')[1].strip()
                    metadata['grid_size'] = size_str
                elif 'Time Steps:' in line:
                    metadata['time_steps'] = int(line.split(':')[1].strip())
                elif 'Final Time:' in line:
                    metadata['final_time'] = float(line.split(':')[1].strip())
                elif 'Threads:' in line:
                    metadata['threads'] = int(line.split(':')[1].strip())
            else:
                break

    # Load the numerical data, skipping header lines
    try:
        data = pd.read_csv(filename, comment='#', header=0)
        # Convert to numpy array
        heat_data = data.values
        print(f"Loaded {filename}: Shape {heat_data.shape}")
        return heat_data, metadata
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None

def calculate_errors(reference_data, comparison_data, method_name):

    if reference_data is None or comparison_data is None:
        return None

    if reference_data.shape != comparison_data.shape:
        print(f"Error: Shape mismatch for {method_name}")
        print(f"Reference shape: {reference_data.shape}, Comparison shape: {comparison_data.shape}")
        return None

    # Calculate point-wise absolute errors
    absolute_errors = np.abs(reference_data - comparison_data)

    # Calculate relative errors (avoid division by zero)
    relative_errors = np.divide(absolute_errors, np.abs(reference_data), 
                               out=np.zeros_like(absolute_errors), 
                               where=np.abs(reference_data)>1e-15)

    # Calculate error metrics
    mse = np.mean(np.square(reference_data - comparison_data))
    rmse = np.sqrt(mse)
    max_abs_error = np.max(absolute_errors)
    mean_abs_error = np.mean(absolute_errors)
    max_rel_error = np.max(relative_errors)
    mean_rel_error = np.mean(relative_errors)

    # Standard deviation of errors
    std_abs_error = np.std(absolute_errors)

    error_metrics = {
        'method': method_name,
        'mse': mse,
        'rmse': rmse,
        'max_absolute_error': max_abs_error,
        'mean_absolute_error': mean_abs_error,
        'max_relative_error': max_rel_error,
        'mean_relative_error': mean_rel_error,
        'std_absolute_error': std_abs_error,
        'absolute_errors': absolute_errors,
        'relative_errors': relative_errors
    }

    return error_metrics

def generate_error_report(all_errors, output_file='heat_error_report.txt'):
    """
    Generate a comprehensive error analysis report.
    FIX: This version is updated to handle 'all_errors' as a dictionary.

    Args:
        all_errors (dict): Dictionary mapping method names to their error data.
        output_file (str): Output file name for the report.
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HEAT EQUATION SOLVER ERROR ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("SUMMARY:\n")
        f.write("This report compares parallel implementations against the serial baseline.\n\n")

        # Create summary table
        f.write("ERROR METRICS COMPARISON:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':<15} {'MSE':<12} {'RMSE':<12} {'Max Abs Err':<12} {'Mean Abs Err':<12}\n")
        f.write("-"*80 + "\n")

        # FIX: Iterate over dictionary items (key, value pairs)
        for method_name, error_data in all_errors.items():
            f.write(f"{method_name:<15} ")
            f.write(f"{error_data['mse']:<12.2e} ")
            f.write(f"{error_data['rmse']:<12.2e} ")
            f.write(f"{error_data['max_absolute_error']:<12.2e} ")
           # f.write(f"{error_data['mean_absolute_error']:<12.2e}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED ANALYSIS:\n")
        f.write("="*80 + "\n\n")

        # FIX: Iterate over dictionary items again for the detailed section
        for method_name, error_data in all_errors.items():
            f.write(f"METHOD: {method_name.upper()}\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean Square Error (MSE):           {error_data['mse']:.6e}\n")
            f.write(f"Root Mean Square Error (RMSE):     {error_data['rmse']:.6e}\n")
            f.write(f"Maximum Absolute Error:            {error_data['max_absolute_error']:.6e}\n")

        f.write("INTERPRETATION GUIDE:\n")
        f.write("...") # The rest of your guide



def create_error_visualizations(all_errors, reference_data):
    """
    Creates and saves visualizations for the heat equation error analysis.
    This corrected version uses imshow for efficient 2D plotting and to prevent
    the 'Image size is too large' error.
    """
    print("Creating corrected error visualizations...")
    
    # Determine the number of plots needed: 1 for the reference solution
    # plus one for each parallel method's error map.
    num_methods = len(all_errors)
    num_plots = 1 + num_methods
    
    # Arrange the plots in a grid, for example, 2 plots per row.
    ncols = 2
    nrows = (num_plots + ncols - 1) // ncols
    
    # Set a reasonable, fixed figure size (width, height in inches).
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows), constrained_layout=True)
    
    # Flatten the axes array for easy iteration, regardless of grid shape.
    axes = np.ravel(axes)

    # --- Plot 1: Reference (Serial) Final Heat Distribution ---
    ax_ref = axes[0]
    im_ref = ax_ref.imshow(reference_data, cmap='hot', aspect='auto')
    ax_ref.set_title('Reference (Serial) Solution')
    ax_ref.set_xlabel('Grid X')
    ax_ref.set_ylabel('Grid Y')
    fig.colorbar(im_ref, ax=ax_ref, label='Temperature')

    # --- Plot Error Maps for Each Parallel Method ---
    plot_index = 1
    for method, data in all_errors.items():
        if plot_index >= len(axes):
            break  # Stop if we run out of subplot axes

        ax_err = axes[plot_index]
        error_map = data.get('absolute_error')
        
        if error_map is not None:
            # Use a diverging colormap (e.g., 'coolwarm') for errors.
            im_err = ax_err.imshow(error_map, cmap='coolwarm', aspect='auto')
            ax_err.set_title(f'Absolute Error: {method.upper()}')
            ax_err.set_xlabel('Grid X')
            ax_err.set_ylabel('Grid Y')
            fig.colorbar(im_err, ax=ax_err, label='Error Magnitude')
        
        plot_index += 1

    # Hide any unused subplots in the grid.
    for i in range(plot_index, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Heat Equation Solver: Error Analysis', fontsize=16)
    
    # Save the generated figure to a file.
    output_filename = 'heat_error_analysis.png'
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved successfully as '{output_filename}'")
    except Exception as e:
        print(f"Failed to save visualization: {e}")
    
    plt.close(fig)  # Close the figure to free up memory.

def main():
    """
    Main function to run the heat equation error analysis.
    This corrected version ensures 'all_errors' is a dictionary.
    """
    print("Heat Equation Solver Error Analysis")
    print("=" * 50)

    # --- Data Loading ---
    try:
        serial_data, serial_meta = load_heat_data('serial_heat_distribution.csv')
        print(f"Loaded serial_heat_distribution.csv: Shape {serial_data.shape}")
        
        openmp_data, _ = load_heat_data('openmp_heat_distribution.csv')
        print(f"Loaded openmp_heat_distribution.csv: Shape {openmp_data.shape}")

        cuda_data, _ = load_heat_data('cuda_heat_distribution.csv')
        print(f"Loaded cuda_heat_distribution.csv: Shape {cuda_data.shape}")
        
        hybrid_data, _ = load_heat_data('hybrid_heat_distribution.csv')
        print(f"Loaded hybrid_heat_distribution.csv: Shape {hybrid_data.shape}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure all heat distribution CSV files are present.")
        return

    reference_data = serial_data
    comparison_methods = {
        "openmp": openmp_data,
        "cuda": cuda_data,
        "hybrid": hybrid_data
    }
    
    print(f"\nReference data shape: {reference_data.shape}\n")

    # --- Error Calculation ---
    # FIX: Initialize 'all_errors' as a dictionary, not a list.
    all_errors = {} 

    for method, data in comparison_methods.items():
        if reference_data.shape != data.shape:
            print(f"Skipping {method}: Shape mismatch {data.shape} vs reference {reference_data.shape}")
            continue
        
        print(f"Calculating errors for {method} method...")
        
        # Calculate error metrics
        absolute_error = np.abs(reference_data - data)
        mse = np.mean((reference_data - data) ** 2)
        rmse = np.sqrt(mse)
        max_abs_error = np.max(absolute_error)
        
        print(f"  MSE: {mse:.6e}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  Max Absolute Error: {max_abs_error:.6e}\n")

        # FIX: Assign the results to a key in the dictionary.
        all_errors[method] = {
            'mse': mse,
            'rmse': rmse,
            'max_absolute_error': max_abs_error,
            'absolute_error': absolute_error
        }

    # --- Reporting and Visualization ---
    print("Generating error analysis report...")
    generate_error_report(all_errors, 'heat_error_report.txt')
    print(f"Report saved as 'heat_error_report.txt'")

    # This function call will now work correctly.
    create_error_visualizations(all_errors, reference_data)

if __name__ == '__main__':
    main()