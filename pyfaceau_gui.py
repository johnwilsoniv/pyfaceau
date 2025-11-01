#!/usr/bin/env python3
"""
PyFaceAU - GUI Interface for Action Unit Extraction

Provides a simple graphical interface for selecting video files and processing
them through the PyFaceAU pipeline.

Usage:
    python pyfaceau_gui.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import threading
from pyfaceau import OpenFaceProcessor


class PyFaceAUGUI:
    """Simple GUI for PyFaceAU video processing"""

    def __init__(self, root):
        self.root = root
        self.root.title("PyFaceAU - Action Unit Extraction")
        self.root.geometry("700x500")

        # Variables
        self.input_files = []
        self.output_dir = None
        self.processor = None
        self.processing = False

        # Create UI
        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets"""

        # Header
        header = tk.Label(
            self.root,
            text="PyFaceAU - Facial Action Unit Extraction",
            font=("Arial", 16, "bold")
        )
        header.pack(pady=20)

        # Info label
        info = tk.Label(
            self.root,
            text="Select video files to extract facial Action Units\n"
                 "92% correlation with OpenFace 2.2 | 72 fps processing speed",
            font=("Arial", 10)
        )
        info.pack(pady=5)

        # Input files section
        input_frame = tk.LabelFrame(self.root, text="Input Videos", padx=10, pady=10)
        input_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # File list
        self.file_listbox = tk.Listbox(input_frame, height=8)
        self.file_listbox.pack(fill="both", expand=True, pady=5)

        # Input buttons
        input_btn_frame = tk.Frame(input_frame)
        input_btn_frame.pack(fill="x", pady=5)

        tk.Button(
            input_btn_frame,
            text="Add Files",
            command=self.add_files,
            width=15
        ).pack(side="left", padx=5)

        tk.Button(
            input_btn_frame,
            text="Clear Files",
            command=self.clear_files,
            width=15
        ).pack(side="left", padx=5)

        # Output directory section
        output_frame = tk.LabelFrame(self.root, text="Output Directory", padx=10, pady=10)
        output_frame.pack(fill="x", padx=20, pady=10)

        self.output_label = tk.Label(
            output_frame,
            text="No directory selected",
            anchor="w",
            fg="gray"
        )
        self.output_label.pack(fill="x", pady=5)

        tk.Button(
            output_frame,
            text="Select Output Directory",
            command=self.select_output_dir,
            width=25
        ).pack(pady=5)

        # Options
        options_frame = tk.LabelFrame(self.root, text="Options", padx=10, pady=10)
        options_frame.pack(fill="x", padx=20, pady=10)

        self.use_clnf = tk.BooleanVar(value=True)
        tk.Checkbutton(
            options_frame,
            text="Use CLNF landmark refinement (recommended for accuracy)",
            variable=self.use_clnf
        ).pack(anchor="w")

        # Progress section
        progress_frame = tk.Frame(self.root, padx=10, pady=10)
        progress_frame.pack(fill="x", padx=20)

        self.progress_label = tk.Label(
            progress_frame,
            text="Ready to process",
            anchor="w"
        )
        self.progress_label.pack(fill="x", pady=5)

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='indeterminate'
        )
        self.progress_bar.pack(fill="x", pady=5)

        # Process button
        self.process_btn = tk.Button(
            self.root,
            text="Process Videos",
            command=self.process_videos,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2,
            width=20
        )
        self.process_btn.pack(pady=20)

    def add_files(self):
        """Add video files to process"""
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv"),
                ("All files", "*.*")
            ]
        )

        for file in files:
            if file not in self.input_files:
                self.input_files.append(file)
                self.file_listbox.insert(tk.END, Path(file).name)

    def clear_files(self):
        """Clear file list"""
        self.input_files = []
        self.file_listbox.delete(0, tk.END)

    def select_output_dir(self):
        """Select output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory"
        )

        if directory:
            self.output_dir = directory
            self.output_label.config(
                text=directory,
                fg="black"
            )

    def process_videos(self):
        """Process selected videos"""

        # Validate inputs
        if not self.input_files:
            messagebox.showerror("Error", "Please select at least one video file")
            return

        if not self.output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return

        if self.processing:
            messagebox.showwarning("Warning", "Processing already in progress")
            return

        # Start processing in background thread
        self.processing = True
        self.process_btn.config(state="disabled")
        self.progress_bar.start()

        thread = threading.Thread(target=self._process_thread, daemon=True)
        thread.start()

    def _process_thread(self):
        """Background processing thread"""
        try:
            # Initialize processor
            self.update_progress("Initializing PyFaceAU pipeline...")

            # Find weights directory
            script_dir = Path(__file__).parent
            weights_dir = script_dir / 'weights'

            if not weights_dir.exists():
                raise FileNotFoundError(
                    f"Weights directory not found: {weights_dir}\n"
                    "Please ensure weights are in the 'weights' subdirectory."
                )

            self.processor = OpenFaceProcessor(
                weights_dir=str(weights_dir),
                use_clnf_refinement=self.use_clnf.get(),
                verbose=False
            )

            # Process each file
            total_files = len(self.input_files)
            successful = 0
            failed = []

            for i, input_file in enumerate(self.input_files, 1):
                try:
                    filename = Path(input_file).name
                    self.update_progress(
                        f"Processing {i}/{total_files}: {filename}"
                    )

                    # Generate output path
                    output_filename = Path(input_file).stem + ".csv"
                    output_path = Path(self.output_dir) / output_filename

                    # Process video
                    frame_count = self.processor.process_video(
                        input_file,
                        str(output_path)
                    )

                    if frame_count > 0:
                        successful += 1
                    else:
                        failed.append((filename, "No frames processed"))

                except Exception as e:
                    failed.append((filename, str(e)))

            # Show results
            self.show_results(successful, failed, total_files)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error",
                f"Failed to initialize pipeline:\n{str(e)}"
            ))

        finally:
            # Reset UI
            self.processing = False
            self.root.after(0, self._reset_ui)

    def update_progress(self, message):
        """Update progress label"""
        self.root.after(0, lambda: self.progress_label.config(text=message))

    def show_results(self, successful, failed, total):
        """Show processing results"""
        message = f"Processing Complete\n\n"
        message += f"Total files: {total}\n"
        message += f"Successful: {successful}\n"
        message += f"Failed: {len(failed)}\n\n"

        if failed:
            message += "Failed files:\n"
            for filename, error in failed[:5]:  # Show first 5 failures
                message += f"- {filename}: {error}\n"
            if len(failed) > 5:
                message += f"... and {len(failed) - 5} more\n"

        message += f"\nOutput directory:\n{self.output_dir}"

        self.root.after(0, lambda: messagebox.showinfo("Results", message))

    def _reset_ui(self):
        """Reset UI after processing"""
        self.progress_bar.stop()
        self.process_btn.config(state="normal")
        self.progress_label.config(text="Ready to process")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PyFaceAUGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
