import tkinter as tk
import sys
from tkinter import ttk, filedialog
import os
sys.path.append(os.getcwd() +  f"{os.sep}.." )
from preprocess.loader import DataStream
import csv
# Sample posts to simulate your data


ds = DataStream(400, "jobs_db")
resume, jobs = ds.send_data(60)

posts = [(i.get_id(), i.form()) for i in jobs]

class LabelingPostViewer(tk.Tk):
    def __init__(self, posts):
        super().__init__()
        self.title("📘 Post Labeler")
        self.geometry("700x500")
        self.configure(bg="#f0f4f8")
        self.minsize(500, 300)

        self.posts = [p[1] for p in posts]
        self.posts_idx = [p[0] for p in posts]
        self.labels = [None] * len(posts)  # 0 or 1 or None
        self.index = 0
        self.label_var = tk.IntVar(value=-1)

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Title
        tk.Label(self, text="Label Each Post", font=("Helvetica", 20, "bold"),
                 bg="#f0f4f8", fg="#333").grid(row=0, pady=(20, 5))

        # Text Area
        text_frame = tk.Frame(self, bg="#f0f4f8")
        text_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)

        self.text_area = tk.Text(text_frame, wrap=tk.WORD, font=("Helvetica", 14),
                                 bg="#ffffff", fg="#222", relief=tk.FLAT)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.text_area.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_area.config(yscrollcommand=scrollbar.set)

        # Label Options
        label_frame = tk.Frame(self, bg="#f0f4f8")
        label_frame.grid(row=3, pady=5)

        tk.Label(label_frame, text="Label:", font=("Helvetica", 12), bg="#f0f4f8").pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(label_frame, text="👍 1", variable=self.label_var, value=1,
                       bg="#f0f4f8", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=5)

        tk.Radiobutton(label_frame, text="👎 0", variable=self.label_var, value=0,
                       bg="#f0f4f8", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=5)

        ttk.Button(label_frame, text="💾 Save Label", command=self.save_label).pack(side=tk.LEFT, padx=10)

        # Navigation + Export
        bottom_frame = tk.Frame(self, bg="#f0f4f8")
        bottom_frame.grid(row=4, pady=10)

        self.prev_btn = ttk.Button(bottom_frame, text="⏪ Previous", command=self.show_prev)
        self.prev_btn.grid(row=0, column=0, padx=10)

        self.next_btn = ttk.Button(bottom_frame, text="Next ⏩", command=self.show_next)
        self.next_btn.grid(row=0, column=1, padx=10)

        ttk.Button(bottom_frame, text="📤 Export Json", command=self.export_json).grid(row=0, column=2, padx=20)

        self.update_display()

    def update_display(self):
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, self.posts[self.index])
        self.text_area.yview_moveto(0)

        label = self.labels[self.index]
        self.label_var.set(label if label is not None else -1)

        self.prev_btn.config(state=tk.NORMAL if self.index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.index < len(self.posts) - 1 else tk.DISABLED)

    def show_prev(self):
        if self.index > 0:
            self.index -= 1
            self.update_display()

    def show_next(self):
        if self.index < len(self.posts) - 1:
            self.index += 1
            self.update_display()

    def save_label(self):
        val = self.label_var.get()
        if val in (0, 1):
            self.labels[self.index] = val
            print(f"✅ Saved label for post {self.index + 1}: {val}")
        else:
            print("⚠️ Please select a label before saving.")

    def export_json(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("Json files", "*.json")],
                                                 title="Save labeled posts")
        if not file_path:
            return

        try:
            import json 
            data_out = {}
            print("lables=", self.labels[:5])
            for i in range(len(self.labels)):
                if self.labels[i] is not None:
                    data_out[self.posts_idx[i]] = self.labels[i]
            with open(file_path, "w") as f:
                json.dump(data_out, f)
            print(f"📁 Exported to: {file_path}")
        except Exception as e:
            print(f"❌ Failed to export: {e}")

if __name__ == "__main__":
    app = LabelingPostViewer(posts)
    app.mainloop()