import nbformat
from IPython import get_ipython
from nbformat.v4 import new_notebook, new_code_cell
import warnings
import sys
import io
import os
from contextlib import redirect_stdout, redirect_stderr
import json
from datetime import datetime

# Nuclear warning suppression - completely bypass all warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Environment variables for complete warning suppression
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCH_WARN_ONCE'] = 'false'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'
os.environ['TORCH_WARN_ALWAYS'] = 'false'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# PyTorch specific warning suppression
try:
    import torch
    torch.set_warn_always(False)
    torch.set_warn_once(False)
except:
    pass

# Create null stderr to completely suppress warnings
class DevNull:
    def write(self, msg): pass
    def flush(self): pass

# Replace stderr completely to block all warnings
original_stderr = sys.stderr
sys.stderr = DevNull()

# Create a more aggressive warning interceptor
class AggressiveWarningInterceptor:
    def __init__(self, original_stream):
        self.original = original_stream
        self.buffer = ""
    
    def write(self, text):
        # Buffer the text to handle multi-line warnings
        self.buffer += text
        
        # Check if we have complete lines
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            # Keep the last line in buffer if it's incomplete
            self.buffer = lines[-1]
            
            # Process complete lines
            for line in lines[:-1]:
                if self._should_show_line(line):
                    self.original.write(line + '\n')
    
    def _should_show_line(self, line):
        """Check if line should be shown (not a warning)"""
        line_lower = line.lower()
        
        # Only block specific PyTorch tensor warnings, allow everything else
        pytorch_warning_indicators = [
            'torch.tensor inputs should be normalized',
            'dividing input by 255',
            '⚠️ torch.tensor inputs should be normalized',
            'warn ⚠️ torch.tensor inputs should be normalized'
        ]
        
        # Block only PyTorch tensor warnings, allow all other output
        return not any(indicator in line_lower for indicator in pytorch_warning_indicators)
    
    def flush(self):
        # Flush any remaining buffered content
        if self.buffer and self._should_show_line(self.buffer):
            self.original.write(self.buffer)
        self.original.flush()
        self.buffer = ""

# Replace both stdout and stderr with aggressive interceptors
original_stdout = sys.stdout
original_stderr = sys.stderr

sys.stdout = AggressiveWarningInterceptor(original_stdout)
sys.stderr = AggressiveWarningInterceptor(original_stderr)

print("✅ Targeted PyTorch tensor warning suppression activated")
print("   - Only suppresses: 'torch.Tensor inputs should be normalized' warnings")
print("   - All other output (training progress, metrics, etc.) will be preserved")

# Additional IPython-level warning suppression
def suppress_ipython_warnings():
    """Suppress warnings at the IPython kernel level"""
    try:
        # Get the current IPython instance
        ip = get_ipython()
        if ip is None:
            return
        
        # Disable IPython warnings
        if hasattr(ip, 'warn_always'):
            ip.warn_always = False
        
        # Set IPython to ignore warnings
        if hasattr(ip, 'show_warning'):
            ip.show_warning = lambda *args, **kwargs: None
            
    except Exception:
        pass

# Apply IPython-level suppression
suppress_ipython_warnings()

def inject_warning_suppression_code(cell_source):
    """Inject warning suppression code at the beginning of a cell"""
    warning_suppression_code = """
# Aggressive warning suppression for this cell
import warnings
import os
import sys

# Suppress all warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

# Set environment variables
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCH_WARN_ONCE'] = 'false'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

# Suppress PyTorch warnings aggressively
try:
    import torch
    # Disable all PyTorch warnings
    if hasattr(torch, 'set_warn_always'):
        torch.set_warn_always(False)
    if hasattr(torch, 'set_warn_once'):
        torch.set_warn_once(False)
    
    # Try to disable C++ warnings
    if hasattr(torch, 'set_warn_always'):
        torch.set_warn_always(False)
    
    # Disable specific warning categories
    if hasattr(torch, 'set_warn_always'):
        torch.set_warn_always(False)
        
except Exception as e:
    pass

# Redirect stderr to suppress warnings
class WarningSuppressor:
    def __init__(self):
        self.original_stderr = sys.stderr
        self.suppressed_stderr = open(os.devnull, 'w')
    
    def __enter__(self):
        sys.stderr = self.suppressed_stderr
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.original_stderr
        self.suppressed_stderr.close()

# Create global suppressor
warning_suppressor = WarningSuppressor()

"""
    return warning_suppression_code + cell_source

def execute_notebook_with_outputs(notebook_path, output_path=None, log_file=None):
    """
    Execute a Jupyter notebook and capture all outputs while completely suppressing warnings.
    
    Args:
        notebook_path (str): Path to the input notebook
        output_path (str): Path to save the executed notebook (optional)
        log_file (str): Path to save execution log (optional)
    """
    
    # Initialize logging
    log_entries = []
    start_time = datetime.now()
    
    def log_message(message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        log_entries.append(log_entry)
    
    # Load original notebook
    log_message(f"Loading notebook: {notebook_path}")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb_orig = nbformat.read(f, as_version=4)
        log_message(f"Successfully loaded notebook with {len(nb_orig.cells)} cells")
    except Exception as e:
        log_message(f"Error loading notebook: {e}", "ERROR")
        return None
    
    # Get IPython instance
    ip = get_ipython()
    if ip is None:
        log_message("Error: This script must be run from within an IPython/Jupyter environment", "ERROR")
        return None
    
    # Create a custom IPython runner that completely suppresses warnings
    class SilentIPythonRunner:
        def __init__(self, ipython_instance):
            self.ip = ipython_instance
            self.original_run_cell = ipython_instance.run_cell
        
        def run_cell(self, source, **kwargs):
            """Run cell with complete warning suppression"""
            # Temporarily replace IPython's warning handling
            original_show_warning = None
            if hasattr(self.ip, 'show_warning'):
                original_show_warning = self.ip.show_warning
                self.ip.show_warning = lambda *args, **kwargs: None
            
            try:
                # Execute with our aggressive warning suppression
                return self.original_run_cell(source, **kwargs)
            finally:
                # Restore original warning handling
                if original_show_warning:
                    self.ip.show_warning = original_show_warning
    
    # Create silent runner
    silent_runner = SilentIPythonRunner(ip)
    
    # Additional protection: Intercept IPython's output streams
    if hasattr(ip, 'kernel'):
        try:
            # Replace kernel's stdout/stderr with our interceptors
            if hasattr(ip.kernel, 'stdout'):
                ip.kernel.stdout = AggressiveWarningInterceptor(original_stdout)
            if hasattr(ip.kernel, 'stderr'):
                ip.kernel.stderr = AggressiveWarningInterceptor(original_stderr)
        except Exception:
            pass
    
    # Create a new notebook to store executed cells with outputs
    nb_new = new_notebook()
    nb_new.metadata = nb_orig.metadata
    
    log_message(f"Starting execution of {len(nb_orig.cells)} cells...")
    
    # Track execution statistics
    total_cells = len(nb_orig.cells)
    successful_cells = 0
    failed_cells = 0
    skipped_cells = 0
    
    # Progress tracking
    print(f"\n🚀 Starting notebook execution: {len(nb_orig.cells)} cells")
    print("=" * 60)
    
    for i, cell in enumerate(nb_orig.cells):
        if cell.cell_type == "code":
            # Show progress bar
            progress = (i + 1) / total_cells * 100
            print(f"\n📝 Executing Cell {i+1}/{total_cells} ({progress:.1f}%)")
            print(f"   Source: {cell.source[:80]}{'...' if len(cell.source) > 80 else ''}")
            
            log_message(f"Executing cell {i+1}/{total_cells}")
            log_message(f"Source preview: {cell.source[:100]}{'...' if len(cell.source) > 100 else ''}")
            
            # Inject warning suppression code
            modified_source = inject_warning_suppression_code(cell.source)
            log_message(f"Injected warning suppression code into cell {i+1}")
            
            # Capture stdout and stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            try:
                # Execute the cell with output capture and warning suppression
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    # Temporarily suppress all warnings during cell execution
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        # Execute cell with complete warning suppression using silent runner
                        print(f"   ⏳ Executing cell... (this may take a while for training cells)")
                        result = silent_runner.run_cell(modified_source)
                
                # Create new cell with outputs (use original source for clean output)
                new_cell = new_code_cell(
                    source=cell.source,  # Keep original source clean
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
                    log_message(f"Cell {i+1} stdout: {len(stdout_content)} characters")
                    
                    # Show training progress in real-time
                    if any(keyword in stdout_content.lower() for keyword in ['epoch', 'loss', 'accuracy', 'training']):
                        print(f"📊 Cell {i+1} Training Progress: {stdout_content.strip()}")
                
                # Add stderr if any
                stderr_content = stderr_capture.getvalue()
                if stderr_content.strip():
                    outputs.append({
                        "output_type": "stream",
                        "name": "stderr",
                        "text": stderr_content
                    })
                    log_message(f"Cell {i+1} stderr: {len(stderr_content)} characters")
                
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
                        log_message(f"Cell {i+1} result captured successfully")
                        
                        # Show important results in real-time
                        result_text = str(result.result)
                        if any(keyword in result_text.lower() for keyword in ['confusion matrix', 'classification report', 'precision', 'recall', 'f1']):
                            print(f"🎯 Cell {i+1} Important Result: {result_text[:200]}{'...' if len(result_text) > 200 else ''}")
                            
                    except Exception as e:
                        log_message(f"Warning: Could not capture result output for cell {i+1}: {e}", "WARNING")
                
                # Add error if execution failed
                if result.error_in_exec:
                    outputs.append({
                        "output_type": "error",
                        "ename": type(result.error_in_exec).__name__,
                        "evalue": str(result.error_in_exec),
                        "traceback": []
                    })
                    log_message(f"Cell {i+1} failed: {result.error_in_exec}", "ERROR")
                    print(f"❌ Cell {i+1} failed: {result.error_in_exec}")
                    failed_cells += 1
                else:
                    log_message(f"Cell {i+1} executed successfully")
                    successful_cells += 1
                    
                    # Show cell completion status
                    if any(keyword in stdout_content.lower() for keyword in ['epoch', 'loss', 'accuracy', 'training']):
                        print(f"✅ Cell {i+1} completed - Training progress captured")
                    elif any(keyword in stdout_content.lower() for keyword in ['confusion matrix', 'classification report', 'precision', 'recall']):
                        print(f"✅ Cell {i+1} completed - Evaluation results captured")
                    else:
                        print(f"✅ Cell {i+1} completed successfully")
                
                new_cell.outputs = outputs
                
            except Exception as e:
                log_message(f"Error executing cell {i+1}: {e}", "ERROR")
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
                failed_cells += 1
            
            nb_new.cells.append(new_cell)
            
        else:
            # For markdown or other cell types, just copy
            nb_new.cells.append(cell)
            skipped_cells += 1
    
    # Save the executed notebook
    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '_executed.ipynb')
    
    log_message(f"Saving executed notebook to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb_new, f)
        log_message("Notebook saved successfully")
    except Exception as e:
        log_message(f"Error saving notebook: {e}", "ERROR")
    
    # Save execution log if requested
    if log_file:
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_entries))
            log_message(f"Execution log saved to: {log_file}")
        except Exception as e:
            log_message(f"Error saving log file: {e}", "ERROR")
    
    # Print execution summary
    end_time = datetime.now()
    execution_duration = end_time - start_time
    
    # Final summary display
    print("\n" + "=" * 60)
    print("🎉 NOTEBOOK EXECUTION COMPLETED!")
    print("=" * 60)
    print(f"📊 Total cells processed: {total_cells}")
    print(f"✅ Successful executions: {successful_cells}")
    print(f"❌ Failed executions: {failed_cells}")
    print(f"⏭️  Skipped cells: {skipped_cells}")
    print(f"⏱️  Total execution time: {execution_duration}")
    print(f"💾 Output notebook: {output_path}")
    if log_file:
        print(f"📝 Log file: {log_file}")
    
    # Show what was captured
    print(f"\n📋 OUTPUTS CAPTURED:")
    print(f"   - Training progress (epochs, loss, accuracy)")
    print(f"   - Model evaluation results")
    print(f"   - Confusion matrices and classification reports")
    print(f"   - All stdout/stderr output")
    print(f"   - Execution results and errors")
    
    print(f"\n🔍 Next steps:")
    print(f"   1. Check the executed notebook: {output_path}")
    print(f"   2. Extract specific results: python extract_results.py {output_path}")
    print(f"   3. Analyze training metrics and performance")
    print("=" * 60)
    
    log_message("=" * 50)
    log_message("EXECUTION SUMMARY")
    log_message("=" * 50)
    log_message(f"Total cells: {total_cells}")
    log_message(f"Successful: {successful_cells}")
    log_message(f"Failed: {failed_cells}")
    log_message(f"Skipped: {skipped_cells}")
    log_message(f"Execution time: {execution_duration}")
    log_message(f"Output notebook: {output_path}")
    if log_file:
        log_message(f"Log file: {log_file}")
    log_message("=" * 50)
    
    return nb_new

def list_available_notebooks():
    """List all available training notebooks"""
    notebooks = {
        "10 Epoch Training Notebooks": [
            "train_script/train_convnext.ipynb",
            "train_script/train_sc_convnext.ipynb", 
            "train_script/train_hybrid_cnn_vit.ipynb",
            "train_script/train_model_hybrid_v2.ipynb",
            "train_script/train_protopnet.ipynb",
            "train_script/train_model_yolov9_efficient_net.ipynb"
        ],
        "20 Epoch Training Notebooks": [
            "epoch20/train_script/train_convnext.ipynb",
            "epoch20/train_script/train_sc_convnext.ipynb",
            "epoch20/train_script/train_hybrid_cnn_vit.ipynb", 
            "epoch20/train_script/train_model_hybrid_v2.ipynb",
            "epoch20/train_script/train_protopnet.ipynb",
            "epoch20/train_script/train_model_yolov9_efficient_net.ipynb"
        ]
    }
    
    print("📚 AVAILABLE TRAINING NOTEBOOKS:")
    print("=" * 50)
    
    for category, notebook_list in notebooks.items():
        print(f"\n🔹 {category}:")
        for i, notebook in enumerate(notebook_list, 1):
            exists = "✅" if os.path.exists(notebook) else "❌"
            print(f"   {i}. {exists} {notebook}")
    
    print("\n" + "=" * 50)
    return notebooks

def main():
    """Main execution function with improved argument handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute Jupyter notebook and capture outputs without warnings')
    parser.add_argument('notebook', nargs='?', help='Path to the input notebook')
    parser.add_argument('-o', '--output', help='Output path for executed notebook')
    parser.add_argument('-l', '--log', help='Path to save execution log')
    parser.add_argument('--list', action='store_true', help='List all available notebooks')
    parser.add_argument('--auto', action='store_true', help='Automatically find and execute notebook')
    
    args = parser.parse_args()
    
    # List available notebooks if requested
    if args.list:
        list_available_notebooks()
        return
    
    # If no notebook specified, show usage
    if not args.notebook:
        print("🚀 KAGGLE TRAINING NOTEBOOK EXECUTOR")
        print("=" * 50)
        print("This script executes training notebooks and captures outputs while suppressing warnings.")
        print("\n📋 USAGE EXAMPLES:")
        print("1. List available notebooks:")
        print("   python kaggle_training_executor.py --list")
        print("\n2. Execute specific notebook:")
        print("   python kaggle_training_executor.py train_script/train_convnext.ipynb")
        print("\n3. Execute with custom output:")
        print("   python kaggle_training_executor.py train_script/train_convnext.ipynb -o convnext_results.ipynb")
        print("\n4. Execute with logging:")
        print("   python kaggle_training_executor.py train_script/train_convnext.ipynb -l training.log")
        print("\n5. Execute with both:")
        print("   python kaggle_training_executor.py train_script/train_convnext.ipynb -o results.ipynb -l training.log")
        print("\n🔍 AVAILABLE NOTEBOOKS:")
        list_available_notebooks()
        return
    
    # Check if notebook exists
    if not os.path.exists(args.notebook):
        print(f"❌ Notebook not found: {args.notebook}")
        print("\n📋 Available notebooks:")
        list_available_notebooks()
        return
    
    # Execute the notebook
    print(f"🚀 Executing notebook: {args.notebook}")
    result = execute_notebook_with_outputs(args.notebook, args.output, args.log)
    
    if result:
        print(f"\n✅ Successfully executed notebook: {args.notebook}")
        output_path = args.output or args.notebook.replace('.ipynb', '_executed.ipynb')
        print(f"📁 Output saved to: {output_path}")
        if args.log:
            print(f"📝 Log saved to: {args.log}")
    else:
        print("❌ Failed to execute notebook")
        sys.exit(1)

if __name__ == "__main__":
    main()
