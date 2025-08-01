import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import platform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class HeatEquationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heat Equation Solver Comparison")
        self.root.geometry("1400x800")
        
        # Configuration
        self.grid_sizes = ["100x100", "200x200", "300x300", "500x500"]
        self.implementations = ["Serial", "OpenMP", "CUDA", "Hybrid"]
        self.colors = ['#FF9999', '#99FF99', '#9999FF', '#FFCC99']
        
        # Create main frames
        self.control_frame = ttk.Frame(root, padding="10")
        self.control_frame.pack(fill=tk.X)
        
        self.comparison_frame = ttk.Frame(root, padding="10")
        self.comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        self.output_frame = ttk.Frame(root, padding="10")
        self.output_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control widgets
        ttk.Label(self.control_frame, text="Grid Size (Nx, Ny):").grid(row=0, column=0, padx=5)
        
        self.grid_var = tk.StringVar()
        self.grid_combobox = ttk.Combobox(
            self.control_frame, 
            textvariable=self.grid_var,
            values=self.grid_sizes
        )
        self.grid_combobox.grid(row=0, column=1, padx=5)
        self.grid_combobox.current(1)
        
        ttk.Label(self.control_frame, text="Time Steps:").grid(row=0, column=2, padx=5)
        
        self.steps_var = tk.StringVar(value="1000")
        self.steps_entry = ttk.Entry(self.control_frame, textvariable=self.steps_var, width=8)
        self.steps_entry.grid(row=0, column=3, padx=5)
        
        # Create comparison columns with individual run buttons
        self.columns = []
        for i, impl in enumerate(self.implementations):
            col_frame = ttk.Frame(self.comparison_frame, borderwidth=2, relief="groove")
            col_frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            self.comparison_frame.grid_columnconfigure(i, weight=1)
            
            # Title and run button
            title_frame = ttk.Frame(col_frame)
            title_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(title_frame, text=impl, font=('Helvetica', 12, 'bold')).pack(side=tk.LEFT)
            
            run_btn = ttk.Button(
                title_frame,
                text="Run",
                command=lambda impl=impl, idx=i: self.run_solver(impl, idx)
            )
            run_btn.pack(side=tk.RIGHT, padx=5)
            
            # Plot area
            fig = plt.Figure(figsize=(3, 2.5), dpi=80)
            ax = fig.add_subplot(111)
            canvas = FigureCanvasTkAgg(fig, col_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Metrics display
            metrics_frame = ttk.Frame(col_frame)
            metrics_frame.pack(fill=tk.X, padx=5, pady=5)
            
            metrics_text = scrolledtext.ScrolledText(
                metrics_frame,
                wrap=tk.WORD,
                width=30,
                height=8,
                font=('Courier', 10)
            )
            metrics_text.pack(fill=tk.BOTH, expand=True)
            metrics_text.insert(tk.END, f"Click 'Run' to execute {impl}...")
            metrics_text.config(state=tk.DISABLED)
            
            self.columns.append({
                "frame": col_frame,
                "fig": fig,
                "ax": ax,
                "canvas": canvas,
                "metrics_text": metrics_text,
                "last_result": None,
                "run_button": run_btn
            })
        
        # Output console
        self.output_text = scrolledtext.ScrolledText(
            self.output_frame,
            wrap=tk.WORD,
            width=160,
            height=10,
            font=('Courier', 9)
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize variables
        self.grid_size = 200
        
    def run_solver(self, implementation, col_index):
        try:
            grid_size = int(self.grid_var.get().split('x')[0])
            time_steps = self.steps_var.get()
            self.grid_size = grid_size
            
            column = self.columns[col_index]
            
            # Disable button during execution
            column["run_button"].config(state=tk.DISABLED)
            
            # Update UI to show running state
            column["metrics_text"].config(state=tk.NORMAL)
            column["metrics_text"].delete(1.0, tk.END)
            column["metrics_text"].insert(tk.END, f"Running {implementation}...")
            column["metrics_text"].config(state=tk.DISABLED)
            column["canvas"].draw()
            self.root.update()
            
            # Update output console
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)  # Clear previous output for this run
            self.output_text.insert(tk.END, f"Running {implementation} with grid {grid_size}x{grid_size}, {time_steps} steps...\n")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
            
            # Determine which executable to run
            if platform.system() == "Linux":
                cmd = f"./{implementation.lower()}_heat {grid_size} {grid_size} {time_steps}"
            else:
                cmd = f"{implementation.lower()}_heat {grid_size} {grid_size} {time_steps}"
            
            # Run the command
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Update output console
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, f"\n{implementation} output:\n")
            self.output_text.insert(tk.END, result.stdout)
            if result.stderr:
                self.output_text.insert(tk.END, f"\nErrors:\n{result.stderr}")
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
            
            # Parse and visualize results
            self.parse_and_visualize(result.stdout, implementation, col_index)
            
        except subprocess.CalledProcessError as e:
            error_msg = f"\nError running {implementation} solver:\n{e.stderr}"
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, error_msg)
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
            
            column["metrics_text"].config(state=tk.NORMAL)
            column["metrics_text"].delete(1.0, tk.END)
            column["metrics_text"].insert(tk.END, f"{implementation} Failed!\n{error_msg}")
            column["metrics_text"].config(state=tk.DISABLED)
            
            # Show error in plot area too
            column["ax"].clear()
            column["ax"].text(0.5, 0.5, "Execution Failed", 
                             ha='center', va='center', color='red')
            column["canvas"].draw()
            
        except Exception as e:
            error_msg = f"\nUnexpected error with {implementation}: {str(e)}"
            self.output_text.config(state=tk.NORMAL)
            self.output_text.insert(tk.END, error_msg)
            self.output_text.see(tk.END)
            self.output_text.config(state=tk.DISABLED)
            
            column["metrics_text"].config(state=tk.NORMAL)
            column["metrics_text"].delete(1.0, tk.END)
            column["metrics_text"].insert(tk.END, f"{implementation} Error!\n{error_msg}")
            column["metrics_text"].config(state=tk.DISABLED)
            
            column["ax"].clear()
            column["ax"].text(0.5, 0.5, "Unexpected Error", 
                             ha='center', va='center', color='red')
            column["canvas"].draw()
            
        finally:
            # Re-enable the button
            column["run_button"].config(state=tk.NORMAL)
    
    def parse_and_visualize(self, output, implementation, col_index):
        column = self.columns[col_index]
        
        # Parse the output
        lines = output.split('\n')
        metrics = {
            "Implementation": implementation,
            "Threads": "N/A",
            "GridSize": f"{self.grid_size}x{self.grid_size}",
            "TimeSteps": "N/A",
            "Time": "N/A",
            "Throughput": "N/A",
            "CenterValue": "N/A"
        }
        
        for line in lines:
            if "Implementation:" in line:
                metrics["Implementation"] = line.split(":")[1].strip()
            elif "Threads:" in line:
                metrics["Threads"] = line.split(":")[1].strip()
            elif "GridSize:" in line:
                metrics["GridSize"] = line.split(":")[1].strip()
            elif "TimeSteps:" in line:
                metrics["TimeSteps"] = line.split(":")[1].strip()
            elif "Time:" in line:
                metrics["Time"] = line.split(":")[1].strip()
            elif "Throughput:" in line:
                metrics["Throughput"] = line.split(":")[1].strip()
            elif "CenterValue:" in line:
                metrics["CenterValue"] = line.split(":")[1].strip()
        
        # Update the plot
        column["ax"].clear()
        
        try:
            # Create sample heat distribution
            x = np.linspace(-0.5, 0.5, self.grid_size)
            y = np.linspace(-0.5, 0.5, self.grid_size)
            X, Y = np.meshgrid(x, y)
            Z = np.exp(-50*(X**2 + Y**2))
            
            # Plot with implementation-specific color
            c = column["ax"].pcolormesh(X, Y, Z, cmap='hot', shading='auto')
            column["fig"].colorbar(c, ax=column["ax"])
            column["ax"].set_title(f"{implementation}\nCenter: {metrics['CenterValue']}")
        except Exception as e:
            column["ax"].text(0.5, 0.5, "Plot Error", 
                             ha='center', va='center', color='red')
        
        # Update metrics display
        column["metrics_text"].config(state=tk.NORMAL)
        column["metrics_text"].delete(1.0, tk.END)
        
        metric_display = f"{metrics['Implementation']} Results:\n"
        metric_display += "="*30 + "\n"
        metric_display += f"Grid: {metrics['GridSize']}\n"
        metric_display += f"Steps: {metrics['TimeSteps']}\n"
        if metrics['Threads'] != "N/A":
            metric_display += f"Threads: {metrics['Threads']}\n"
        metric_display += f"Time: {metrics['Time']} s\n"
        metric_display += f"Throughput: {metrics['Throughput']} MLUPS\n"
        metric_display += f"Center Temp: {metrics['CenterValue']}\n"
        
        column["metrics_text"].insert(tk.END, metric_display)
        column["metrics_text"].config(state=tk.DISABLED)
        
        # Redraw
        column["canvas"].draw()
        column["last_result"] = metrics

def main():
    root = tk.Tk()
    app = HeatEquationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

    
#   chmod +x compile_all.sh
#   ./compile_all.sh
#   source venv/bin/activate
#   python3 heat_demo.py
