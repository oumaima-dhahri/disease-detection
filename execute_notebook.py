import nbformat
from IPython import get_ipython
from nbformat.v4 import new_notebook, new_code_cell
import warnings
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import os

# Suppress specific warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='.*torch.Tensor inputs should be normalized.*')
warnings.filterwarnings('ignore', message='.*Dividing input by 255.*')

def execute_notebook_with_outputs(notebook_path, output_path=None):
    """
    Execute a Jupyter notebook and capture all outputs including warnings.
    
    Args:
        notebook_path (str): Path to the input notebook
        output_path (str): Path to save the executed notebook (optional)
    """
    
    # Load original notebook
    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_orig = nbformat.read(f, as_version=4)
    
    # Get IPython instance
    ip = get_ipython()
    if ip is None:
        print("Error: This script must be run from within an IPython/Jupyter environment")
        return None
    
    # Create a new notebook to store executed cells with outputs
    nb_new = new_notebook()
    nb_new.metadata = nb_orig.metadata
    
    print(f"Executing {len(nb_orig.cells)} cells...")
    
    for i, cell in enumerate(nb_orig.cells):
        if cell.cell_type == "code":
            print(f"\n>>> Executing cell {i+1}/{len(nb_orig.cells)}")
            print(f"Source: {cell.source[:100]}{'...' if len(cell.source) > 100 else ''}")
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            try:
                # Execute the cell with output capture
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = ip.run_cell(cell.source)
                
                # Create new cell with outputs
                new_cell = new_code_cell(
                    source=cell.source,
                    execution_count=ip.execution_count
                )
                
                # Capture outputs
                outputs = []
                
                # Add stdout if any
                stdout_content = stdout_capture.getvalue()
                if stdout_content.strip():
                    outputs.append({
                        "output_type": "stream",
                        "name": "stdout",
                        "text": stdout_content
                    })
                
                # Add stderr if any
                stderr_content = stderr_capture.getvalue()
                if stderr_content.strip():
                    outputs.append({
                        "output_type": "stream",
                        "name": "stderr",
                        "text": stderr_content
                    })
                
                # Add execution result if available
                if hasattr(result, 'result') and result.result is not None:
                    try:
                        # Try to get the result as a displayable output
                        if hasattr(result.result, '_repr_html_'):
                            outputs.append({
                                "output_type": "execute_result",
                                "execution_count": ip.execution_count,
                                "data": {"text/html": result.result._repr_html_()},
                                "metadata": {}
                            })
                        elif hasattr(result.result, '_repr_pretty_'):
                            outputs.append({
                                "output_type": "execute_result",
                                "execution_count": ip.execution_count,
                                "data": {"text/plain": str(result.result)},
                                "metadata": {}
                            })
                        else:
                            outputs.append({
                                "output_type": "execute_result",
                                "execution_count": ip.execution_count,
                                "data": {"text/plain": str(result.result)},
                                "metadata": {}
                            })
                    except Exception as e:
                        print(f"Warning: Could not capture result output: {e}")
                
                # Add error if execution failed
                if result.error_in_exec:
                    outputs.append({
                        "output_type": "error",
                        "ename": type(result.error_in_exec).__name__,
                        "evalue": str(result.error_in_exec),
                        "traceback": []
                    })
                    print(f"❌ Cell {i+1} failed: {result.error_in_exec}")
                else:
                    print(f"✅ Cell {i+1} executed successfully")
                
                new_cell.outputs = outputs
                
            except Exception as e:
                print(f"❌ Error executing cell {i+1}: {e}")
                # Create error cell
                new_cell = new_code_cell(
                    source=cell.source,
                    execution_count=ip.execution_count,
                    outputs=[{
                        "output_type": "error",
                        "ename": type(e).__name__,
                        "evalue": str(e),
                        "traceback": []
                    }]
                )
            
            nb_new.cells.append(new_cell)
            
        else:
            # For markdown or other cell types, just copy
            nb_new.cells.append(cell)
    
    # Save the executed notebook
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    
    print(f"\nSaving executed notebook to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb_new, f)
    
    print("✅ Notebook execution completed!")
    return nb_new

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute Jupyter notebook and capture outputs')
    parser.add_argument('notebook', help='Path to the input notebook')
    parser.add_argument('-o', '--output', help='Output path for executed notebook')
    
    args = parser.parse_args()
    
    # Execute the notebook
    result = execute_notebook_with_outputs(args.notebook, args.output)
    
    if result:
        print(f"Successfully executed notebook: {args.notebook}")
        print(f"Output saved to: {args.output or args.notebook.replace('.ipynb', '_executed.ipynb')}")
    else:
        print("Failed to execute notebook")
        sys.exit(1)

if __name__ == "__main__":
    # If run directly, try to execute a default notebook
    try:
        # Try to find and execute the YOLO training notebook
        notebook_path = "train_model_yolov9_efficient_net.ipynb"
        if not os.path.exists(notebook_path):
            print(f"Notebook {notebook_path} not found.")
            print("Usage: python execute_notebook.py <notebook_path>")
            sys.exit(1)
        
        execute_notebook_with_outputs(notebook_path)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Usage: python execute_notebook.py <notebook_path>")
        sys.exit(1)
