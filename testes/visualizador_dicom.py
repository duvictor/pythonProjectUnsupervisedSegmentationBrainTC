import tkinter as tk
from tkinter import ttk
import numpy as np
import pydicom
import os
import glob
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

DICOM_DIR = r"D:\Users\paulo\PycharmProjects\pythonProjectUnsupervisedSegmentationBrainTC\dataset\MR-MS-new-janelado\P0008"


def load_dicom_series(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.dcm")), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))
    slices = []
    for f in files:
        ds = pydicom.dcmread(f)
        slices.append(ds)
    pixel_data = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    return slices, pixel_data


def apply_window(pixel_array, center, width):
    low = center - width / 2
    high = center + width / 2
    windowed = np.clip(pixel_array, low, high)
    windowed = (windowed - low) / (high - low)
    return windowed


class DicomViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador DICOM com Janelamento")
        self.root.configure(bg="#1e1e1e")

        # --- Carregamento ---
        status_label = tk.Label(root, text="Carregando exame...", bg="#1e1e1e", fg="white", font=("Arial", 12))
        status_label.pack(pady=10)
        root.update()

        self.slices, self.pixel_data = load_dicom_series(DICOM_DIR)
        self.n_slices = self.pixel_data.shape[0]

        # Valores padrão de janela a partir do primeiro slice
        ds0 = self.slices[0]
        default_wc = float(ds0.get("WindowCenter", 400))
        default_ww = float(ds0.get("WindowWidth", 800))
        if isinstance(default_wc, pydicom.multival.MultiValue):
            default_wc = float(default_wc[0])
        if isinstance(default_ww, pydicom.multival.MultiValue):
            default_ww = float(default_ww[0])

        global_min = float(self.pixel_data.min())
        global_max = float(self.pixel_data.max())
        self.global_min = global_min
        self.global_max = global_max

        status_label.destroy()

        # --- Layout principal ---
        main_frame = tk.Frame(root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Painel esquerdo: imagem
        left_frame = tk.Frame(main_frame, bg="#1e1e1e")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(6, 6), facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.im = self.ax.imshow(
            apply_window(self.pixel_data[self.n_slices // 2], default_wc, default_ww),
            cmap="gray", vmin=0, vmax=1, interpolation="bilinear"
        )
        self.fig.tight_layout(pad=0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Painel direito: controles
        right_frame = tk.Frame(main_frame, bg="#2a2a2a", width=260)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Controles", bg="#2a2a2a", fg="white",
                 font=("Arial", 13, "bold")).pack(pady=(15, 5))

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=5)

        # --- Slice ---
        tk.Label(right_frame, text="Slice", bg="#2a2a2a", fg="#aaaaaa", font=("Arial", 10)).pack(anchor="w", padx=15)
        self.slice_var = tk.IntVar(value=self.n_slices // 2)
        self.slice_label = tk.Label(right_frame, text=f"{self.n_slices // 2 + 1} / {self.n_slices}",
                                    bg="#2a2a2a", fg="white", font=("Arial", 10))
        self.slice_label.pack(anchor="e", padx=15)
        self.slice_slider = tk.Scale(right_frame, from_=0, to=self.n_slices - 1,
                                     orient=tk.HORIZONTAL, variable=self.slice_var,
                                     command=self._on_change, bg="#2a2a2a", fg="white",
                                     troughcolor="#444", highlightthickness=0, showvalue=False,
                                     length=220)
        self.slice_slider.pack(padx=15, pady=(0, 10))

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=5)

        # --- Window Center ---
        tk.Label(right_frame, text="Window Center (WC)", bg="#2a2a2a", fg="#aaaaaa", font=("Arial", 10)).pack(anchor="w", padx=15)
        self.wc_var = tk.DoubleVar(value=default_wc)
        self.wc_label = tk.Label(right_frame, text=f"{default_wc:.0f}",
                                 bg="#2a2a2a", fg="white", font=("Arial", 10))
        self.wc_label.pack(anchor="e", padx=15)
        wc_range = global_max - global_min
        self.wc_slider = tk.Scale(right_frame, from_=global_min, to=global_max,
                                  orient=tk.HORIZONTAL, variable=self.wc_var,
                                  command=self._on_change, bg="#2a2a2a", fg="white",
                                  troughcolor="#444", highlightthickness=0, showvalue=False,
                                  resolution=1, length=220)
        self.wc_slider.pack(padx=15, pady=(0, 10))

        # --- Window Width ---
        tk.Label(right_frame, text="Window Width (WW)", bg="#2a2a2a", fg="#aaaaaa", font=("Arial", 10)).pack(anchor="w", padx=15)
        self.ww_var = tk.DoubleVar(value=default_ww)
        self.ww_label = tk.Label(right_frame, text=f"{default_ww:.0f}",
                                 bg="#2a2a2a", fg="white", font=("Arial", 10))
        self.ww_label.pack(anchor="e", padx=15)
        self.ww_slider = tk.Scale(right_frame, from_=1, to=wc_range * 2,
                                  orient=tk.HORIZONTAL, variable=self.ww_var,
                                  command=self._on_change, bg="#2a2a2a", fg="white",
                                  troughcolor="#444", highlightthickness=0, showvalue=False,
                                  resolution=1, length=220)
        self.ww_slider.pack(padx=15, pady=(0, 10))

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=5)

        # Botão reset
        tk.Button(right_frame, text="Resetar Janela", command=self._reset_window,
                  bg="#3a3a3a", fg="white", relief=tk.FLAT, font=("Arial", 10),
                  activebackground="#555", cursor="hand2").pack(pady=8, padx=15, fill=tk.X)

        # Info do slice atual
        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=10, pady=5)
        self.info_label = tk.Label(right_frame, text="", bg="#2a2a2a", fg="#888888",
                                   font=("Arial", 9), justify=tk.LEFT, wraplength=230)
        self.info_label.pack(anchor="w", padx=15, pady=5)

        self._default_wc = default_wc
        self._default_ww = default_ww

        self._update_image()
        self._update_info()

        # Teclado: setas para navegar entre slices
        root.bind("<Left>", lambda e: self._step_slice(-1))
        root.bind("<Right>", lambda e: self._step_slice(1))
        root.bind("<Up>", lambda e: self._step_slice(1))
        root.bind("<Down>", lambda e: self._step_slice(-1))

    def _on_change(self, *_):
        self._update_image()
        self._update_info()

    def _step_slice(self, delta):
        new_val = max(0, min(self.n_slices - 1, self.slice_var.get() + delta))
        self.slice_var.set(new_val)
        self._update_image()
        self._update_info()

    def _reset_window(self):
        self.wc_var.set(self._default_wc)
        self.ww_var.set(self._default_ww)
        self._update_image()
        self._update_info()

    def _update_image(self):
        idx = self.slice_var.get()
        wc = self.wc_var.get()
        ww = max(1, self.ww_var.get())
        windowed = apply_window(self.pixel_data[idx], wc, ww)
        self.im.set_data(windowed)
        self.canvas.draw_idle()
        self.slice_label.config(text=f"{idx + 1} / {self.n_slices}")
        self.wc_label.config(text=f"{wc:.0f}")
        self.ww_label.config(text=f"{ww:.0f}")

    def _update_info(self):
        idx = self.slice_var.get()
        ds = self.slices[idx]
        patient = str(ds.get("PatientID", "N/A"))
        modality = str(ds.get("Modality", "N/A"))
        rows = str(ds.get("Rows", "N/A"))
        cols = str(ds.get("Columns", "N/A"))
        spacing = ds.get("PixelSpacing", ["N/A", "N/A"])
        info = (
            f"Paciente: {patient}\n"
            f"Modalidade: {modality}\n"
            f"Tamanho: {rows} x {cols}\n"
            f"Pixel Spacing: {spacing[0]:.2f} x {spacing[1]:.2f} mm\n"
            f"Min global: {self.global_min:.0f}\n"
            f"Max global: {self.global_max:.0f}"
        )
        self.info_label.config(text=info)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x620")
    app = DicomViewer(root)
    root.mainloop()
