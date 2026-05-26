"""
Pipeline de Segmentação de RM Cerebral
=======================================
Etapas:
  1. Carregamento DICOM → volume 3D
  2. Correção de inhomogeneidade (N4 Bias Field Correction via SimpleITK)
  3. Extração cerebral / skull stripping (Otsu + morfologia)
  4. Normalização de intensidade (Z-score dentro da máscara cerebral)
  5. Registro espacial (alinhamento ao volume de referência, se múltiplas sequências)
  6. Segmentação de tecidos (K-means: fundo, LCR, substância cinzenta, substância branca)
  7. Detecção de lesões / hemorragias (outliers de intensidade + componentes conectados)

Saída: Visualizador interativo com overlays de segmentação e lesões por slice.

Dependências:
  pip install pydicom numpy scipy scikit-learn scikit-image matplotlib
  pip install SimpleITK   ← opcional, necessário para N4 bias correction
"""

import os
import glob
import threading
import numpy as np
import pydicom
from scipy import ndimage
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings

warnings.filterwarnings("ignore")

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False

# ── Configurações ─────────────────────────────────────────────────────
DICOM_ROOT         = r"D:\Users\paulo\PycharmProjects\pythonProjectUnsupervisedSegmentationBrainTC\dataset\MR-MS-new"
CACHE_DIR          = r"D:\Users\paulo\PycharmProjects\pythonProjectUnsupervisedSegmentationBrainTC\pipeline\cache"
N_TISSUE_CLASSES   = 4        # FLAIR: CSF(suprimido), GM, WM normal, WM hiperintenso
ANOMALY_Z_THRESH   = 2.0      # FLAIR: threshold mais sensível (lesões EM, edema, hemorragia subaguda)
ANOMALY_WM_STD     = 1.5      # FLAIR: voxels de WM acima de WM_mean + N*std → lesão intra-WM
ANOMALY_MIN_VOXELS = 5        # FLAIR: lesões de EM podem ser pequenas (~3mm), filtro menor
# ─────────────────────────────────────────────────────────────────────

# ── Paleta de cores (RGBA) — adaptada para FLAIR ─────────────────────
# FLAIR: CSF suprimido (escuro), GM moderado, WM mais brilhante, lesões muito brilhantes
# Label 0 = Fundo/CSF  → transparente
# Label 1 = CSF suprimido → azul escuro
# Label 2 = SC (GM)    → verde
# Label 3 = SB (WM)    → amarelo
# Label 4 = WM hiperintenso (FLAIR K-means 4º cluster) → laranja
TISSUE_COLORS = [
    (0.00, 0.00, 0.00, 0.00),   # 0: Fundo
    (0.10, 0.25, 0.75, 0.60),   # 1: CSF suprimido (FLAIR: escuro)
    (0.25, 0.75, 0.25, 0.65),   # 2: Substância Cinzenta
    (0.95, 0.90, 0.20, 0.65),   # 3: Substância Branca normal
    (0.95, 0.55, 0.05, 0.70),   # 4: WM hiperintenso / lesões FLAIR
]
ANOMALY_RGBA   = (0.95, 0.08, 0.08, 0.88)   # vermelho vivo para lesões confirmadas
TISSUE_NAMES   = ["Fundo", "CSF / Líquor (suprimido)", "Substância Cinzenta",
                  "Substância Branca", "WM Hiperintenso"]
LEGEND_ENTRIES = [
    ("#1a3fcc", "CSF / Líquor (suprimido)"),
    ("#44cc44", "Substância Cinzenta"),
    ("#eeee22", "Substância Branca"),
    ("#f08c0a", "WM Hiperintenso (FLAIR)"),
    ("#ee2222", "Lesões / Anomalias"),
]
# ─────────────────────────────────────────────────────────────────────


# ═════════════════════════════════════════════════════════════════════
#  FUNÇÕES DE PIPELINE
# ═════════════════════════════════════════════════════════════════════

def load_dicom_volume(directory):
    """Carrega série DICOM e devolve (volume 3D float32, lista de datasets)."""
    def sort_key(f):
        base = os.path.splitext(os.path.basename(f))[0]
        return int(base) if base.isdigit() else base

    files = sorted(glob.glob(os.path.join(directory, "*.dcm")), key=sort_key)
    if not files:
        raise FileNotFoundError(f"Nenhum .dcm encontrado em: {directory}")

    slices, metas = [], []
    for f in files:
        ds = pydicom.dcmread(f)
        slices.append(ds.pixel_array.astype(np.float32))
        metas.append(ds)

    return np.stack(slices, axis=0), metas


def n4_bias_correction(volume: np.ndarray) -> np.ndarray:
    """
    N4 Bias Field Correction (SimpleITK).
    Corrige inhomogeneidades de campo magnético que causam variações
    suaves de intensidade no volume.
    Se SimpleITK não estiver instalado, devolve o volume sem alteração.
    """
    if not HAS_SITK:
        return volume

    img = sitk.GetImageFromArray(volume.astype(np.float32))
    img = sitk.Cast(img, sitk.sitkFloat32)

    # Máscara binária via Otsu para guiar a correção
    mask = sitk.OtsuThreshold(img, 0, 1, 200)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    corrected = corrector.Execute(img, mask)

    return sitk.GetArrayFromImage(corrected)


def extract_brain_mask(volume: np.ndarray) -> np.ndarray:
    """
    Skull stripping simplificado:
    1. Threshold Otsu separa tecido mole do fundo
    2. Fechamento morfológico 3D preenche lacunas internas
    3. Preenchimento de buracos (fill_holes)
    4. Mantém apenas o maior componente conectado (cérebro)
    5. Erosão leve remove bordas do crânio e artefatos superficiais
    Retorna máscara booleana 3D.
    """
    thresh = threshold_otsu(volume)
    binary = volume > thresh

    struct3d = ndimage.generate_binary_structure(3, 2)
    binary = ndimage.binary_closing(binary, structure=struct3d, iterations=4)
    binary = ndimage.binary_fill_holes(binary)

    labeled, n_comp = ndimage.label(binary)
    if n_comp > 1:
        sizes = ndimage.sum(binary, labeled, range(1, n_comp + 1))
        largest = int(np.argmax(sizes)) + 1
        binary = labeled == largest

    binary = ndimage.binary_erosion(binary, iterations=3)
    return binary.astype(bool)


def normalize_zscore(volume: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """
    Normalização Z-score dentro da máscara cerebral.
    Voxels fora da máscara ficam com valor 0.
    """
    brain_vals = volume[brain_mask]
    mu, sigma = brain_vals.mean(), brain_vals.std()
    if sigma == 0:
        return volume.copy()

    norm = np.zeros_like(volume, dtype=np.float32)
    norm[brain_mask] = (volume[brain_mask] - mu) / sigma
    return norm


def register_to_reference(volume: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Registro espacial: alinha 'volume' ao 'reference' usando transformação
    afim (SimpleITK). Usado quando há múltiplas sequências do mesmo paciente
    (T1, T2, FLAIR) para garantir correspondência voxel-a-voxel.
    Se SimpleITK não estiver disponível, devolve volume sem alteração.
    """
    if not HAS_SITK:
        return volume

    fixed   = sitk.GetImageFromArray(reference.astype(np.float32))
    moving  = sitk.GetImageFromArray(volume.astype(np.float32))

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    ))

    transform  = reg.Execute(fixed, moving)
    resampled  = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0)
    return sitk.GetArrayFromImage(resampled)


def segment_tissues(volume_norm: np.ndarray, brain_mask: np.ndarray,
                    n_classes: int = N_TISSUE_CLASSES) -> np.ndarray:
    """
    Segmentação de tecidos via K-means.
    Clusters ordenados por intensidade média crescente:
      label 1 = LCR (mais escuro), 2 = SC, 3 = SB (mais brilhante).
    Label 0 reservado para fundo (fora da máscara cerebral).
    """
    brain_voxels = volume_norm[brain_mask].reshape(-1, 1)

    km = KMeans(n_clusters=n_classes, random_state=42, n_init=15)
    km.fit(brain_voxels)

    # Reordena labels por intensidade média crescente
    centers = km.cluster_centers_.flatten()
    order   = np.argsort(centers)                  # índice original → posição na ordem
    remap   = np.zeros(n_classes, dtype=np.uint8)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl + 1               # +1: label 0 = fundo

    labels_brain = km.labels_
    remapped     = remap[labels_brain]

    seg_map = np.zeros(volume_norm.shape, dtype=np.uint8)
    seg_map[brain_mask] = remapped
    return seg_map


def detect_anomalies_flair(volume_norm: np.ndarray, brain_mask: np.ndarray,
                           seg_map: np.ndarray,
                           z_thresh: float = ANOMALY_Z_THRESH,
                           wm_std_thresh: float = ANOMALY_WM_STD,
                           min_voxels: int = ANOMALY_MIN_VOXELS) -> np.ndarray:
    """
    Detecção de lesões otimizada para FLAIR.

    Estratégia dupla:
      A) Outlier global — voxels cerebrais com Z-score > z_thresh (2.0).
         Captura hiperintensidades perivasculares, edema, hemorragias subagudas.

      B) Detector intra-WM — dentro da substância branca (label 3), identifica
         voxels com intensidade > WM_mean + wm_std_thresh * WM_std.
         Captura especificamente placas de EM perivasculares que podem não
         atingir o threshold global mas estão claramente acima do WM local.

    A detecção final é a união (A ∪ B), seguida de filtragem por tamanho mínimo
    para descartar ruído pontual.

    Referência clínica FLAIR:
      - Placas de EM: hiperintensidades periventriculares, juxtacorticais, infratentoriais
      - Hemorragia subaguda: isointensa a hiperintensa em FLAIR
      - Edema vasogênico: hiperintenso ao redor de lesões expansivas
    """
    # A) Outlier global por Z-score
    global_outliers = (volume_norm > z_thresh) & brain_mask

    # B) Hiperintensidades dentro da WM (label 3 = SB, o cluster mais brilhante normal)
    wm_mask = seg_map == 3
    wm_intra = np.zeros_like(brain_mask)
    if wm_mask.any():
        wm_vals = volume_norm[wm_mask]
        wm_mean, wm_std = wm_vals.mean(), wm_vals.std()
        wm_intra = (volume_norm > wm_mean + wm_std_thresh * wm_std) & brain_mask

    anomaly = global_outliers | wm_intra

    # Filtragem morfológica: remove componentes menores que min_voxels
    labeled, n = ndimage.label(anomaly)
    if n > 0:
        sizes = ndimage.sum(anomaly, labeled, range(1, n + 1))
        for comp_id, sz in enumerate(sizes, start=1):
            if sz < min_voxels:
                anomaly[labeled == comp_id] = False

    return anomaly


def run_pipeline(patient_dir: str, progress_cb=None) -> dict:
    """
    Executa o pipeline completo para um paciente.
    progress_cb(mensagem: str, percentual: int) → callback opcional de progresso.
    Retorna dicionário com todos os artefatos gerados.
    """
    def cb(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    cb("Carregando DICOMs...", 5)
    volume, metas = load_dicom_volume(patient_dir)

    if HAS_SITK:
        cb("N4 Bias Field Correction (pode demorar alguns minutos)...", 15)
    else:
        cb("N4 ignorado — instale SimpleITK para habilitá-lo...", 15)
    volume_corrected = n4_bias_correction(volume)

    cb("Extração cerebral (skull stripping)...", 40)
    brain_mask = extract_brain_mask(volume_corrected)

    cb("Normalização Z-score...", 55)
    volume_norm = normalize_zscore(volume_corrected, brain_mask)

    cb("Segmentação de tecidos FLAIR (K-means 4 classes)...", 70)
    seg_map = segment_tissues(volume_norm, brain_mask)

    cb("Detecção de lesões FLAIR (Z-score global + intra-WM)...", 88)
    anomaly_mask = detect_anomalies_flair(volume_norm, brain_mask, seg_map)

    cb("Concluído!", 100)

    return {
        "volume":       volume,
        "volume_norm":  volume_norm,
        "brain_mask":   brain_mask,
        "seg_map":      seg_map,
        "anomaly_mask": anomaly_mask,
        "n_slices":     volume.shape[0],
        "metas":        metas,
        "patient_dir":  patient_dir,
    }


# ═════════════════════════════════════════════════════════════════════
#  INTERFACE GRÁFICA
# ═════════════════════════════════════════════════════════════════════

class SegmentationViewer:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.result = None
        self.root.title("Pipeline de Segmentação RM Cerebral")
        self.root.configure(bg="#161616")
        self.root.geometry("1160x740")
        self._build_ui()

    # ── Construção da UI ─────────────────────────────────────────────

    def _build_ui(self):
        self._build_top_bar()
        self._build_body()
        self.root.bind("<Left>",  lambda e: self._step_slice(-1))
        self.root.bind("<Right>", lambda e: self._step_slice(1))
        self.root.bind("<Up>",    lambda e: self._step_slice(1))
        self.root.bind("<Down>",  lambda e: self._step_slice(-1))

    def _build_top_bar(self):
        bar = tk.Frame(self.root, bg="#252525", pady=8)
        bar.pack(fill=tk.X)

        tk.Label(bar, text="Paciente:", bg="#252525", fg="white",
                 font=("Arial", 11)).pack(side=tk.LEFT, padx=(14, 4))

        patient_dirs = sorted([
            d for d in os.listdir(DICOM_ROOT)
            if os.path.isdir(os.path.join(DICOM_ROOT, d))
        ])
        self.patient_var = tk.StringVar(value=patient_dirs[0] if patient_dirs else "")
        ttk.Combobox(bar, textvariable=self.patient_var, values=patient_dirs,
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        self.process_btn = tk.Button(
            bar, text="▶  Processar Pipeline",
            command=self._start_processing,
            bg="#1565c0", fg="white", relief=tk.FLAT,
            font=("Arial", 10, "bold"), padx=12, cursor="hand2",
            activebackground="#0d47a1"
        )
        self.process_btn.pack(side=tk.LEFT, padx=10)

        self.status_lbl = tk.Label(
            bar, text="Selecione um paciente e clique em Processar",
            bg="#252525", fg="#888", font=("Arial", 10)
        )
        self.status_lbl.pack(side=tk.LEFT, padx=8)

        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(bar, variable=self.progress_var,
                        maximum=100, length=180).pack(side=tk.LEFT, padx=8)

        sitk_lbl = ("SimpleITK ativo  ✓" if HAS_SITK
                    else "SimpleITK não instalado — N4 desativado")
        sitk_color = "#44cc44" if HAS_SITK else "#f0a020"
        tk.Label(bar, text=sitk_lbl, bg="#252525", fg=sitk_color,
                 font=("Arial", 9)).pack(side=tk.RIGHT, padx=14)

    def _build_body(self):
        body = tk.Frame(self.root, bg="#161616")
        body.pack(fill=tk.BOTH, expand=True)

        # ── Imagens ──────────────────────────────────────────────────
        img_frame = tk.Frame(body, bg="#161616")
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.fig = Figure(figsize=(9.5, 5.8), facecolor="#161616")
        self.ax_orig = self.fig.add_subplot(121)
        self.ax_seg  = self.fig.add_subplot(122)
        for ax in (self.ax_orig, self.ax_seg):
            ax.set_facecolor("black")
            ax.axis("off")
        self.ax_orig.set_title("Original", color="white", fontsize=10)
        self.ax_seg.set_title("Segmentação + Lesões", color="white", fontsize=10)
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=img_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Painel de controles ──────────────────────────────────────
        ctrl = tk.Frame(body, bg="#222222", width=240)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
        ctrl.pack_propagate(False)

        def section_label(text):
            ttk.Separator(ctrl, orient="horizontal").pack(fill=tk.X, padx=8, pady=(8, 2))
            tk.Label(ctrl, text=text, bg="#222222", fg="#aaaaaa",
                     font=("Arial", 9, "bold")).pack(anchor="w", padx=12)

        tk.Label(ctrl, text="Controles", bg="#222222", fg="white",
                 font=("Arial", 12, "bold")).pack(pady=(14, 0))

        # Slice
        section_label("SLICE")
        self.slice_var = tk.IntVar(value=0)
        self.slice_lbl = tk.Label(ctrl, text="—", bg="#222222", fg="white", font=("Arial", 10))
        self.slice_lbl.pack(anchor="e", padx=12)
        self.slice_slider = tk.Scale(
            ctrl, from_=0, to=1, orient=tk.HORIZONTAL,
            variable=self.slice_var, command=self._update_display,
            bg="#222222", fg="white", troughcolor="#444",
            highlightthickness=0, showvalue=False, length=212
        )
        self.slice_slider.pack(padx=12)

        # Opacidade
        section_label("OPACIDADE DO OVERLAY")
        self.opacity_var = tk.DoubleVar(value=0.55)
        tk.Scale(
            ctrl, from_=0.0, to=1.0, orient=tk.HORIZONTAL, resolution=0.05,
            variable=self.opacity_var, command=self._update_display,
            bg="#222222", fg="white", troughcolor="#444",
            highlightthickness=0, showvalue=False, length=212
        ).pack(padx=12)

        # Camadas
        section_label("CAMADAS VISÍVEIS")
        self.show_seg_var  = tk.BooleanVar(value=True)
        self.show_anom_var = tk.BooleanVar(value=True)
        for var, text in [(self.show_seg_var, "Segmentação de tecidos"),
                          (self.show_anom_var, "Lesões / Anomalias")]:
            tk.Checkbutton(
                ctrl, text=text, variable=var,
                command=self._update_display,
                bg="#222222", fg="white", selectcolor="#333",
                activebackground="#222222", font=("Arial", 10)
            ).pack(anchor="w", padx=12, pady=2)

        # Legenda
        section_label("LEGENDA")
        for hex_color, name in LEGEND_ENTRIES:
            row = tk.Frame(ctrl, bg="#222222")
            row.pack(anchor="w", padx=12, pady=1)
            tk.Label(row, text="■", bg="#222222", fg=hex_color,
                     font=("Arial", 13)).pack(side=tk.LEFT)
            tk.Label(row, text=f" {name}", bg="#222222", fg="#cccccc",
                     font=("Arial", 9)).pack(side=tk.LEFT)

        # Info
        section_label("INFO DO SLICE")
        self.info_lbl = tk.Label(
            ctrl, text="", bg="#222222", fg="#777777",
            font=("Arial", 8), justify=tk.LEFT, wraplength=218
        )
        self.info_lbl.pack(anchor="w", padx=12, pady=4)

    # ── Lógica de processamento ──────────────────────────────────────

    def _start_processing(self):
        patient = self.patient_var.get()
        if not patient:
            return
        patient_dir = os.path.join(DICOM_ROOT, patient)
        self.process_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)

        def worker():
            def cb(msg, pct):
                self.root.after(0, lambda m=msg, p=pct: (
                    self.status_lbl.config(text=m),
                    self.progress_var.set(p)
                ))
            try:
                result = run_pipeline(patient_dir, progress_cb=cb)
                self.root.after(0, lambda: self._on_done(result))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, result):
        self.result = result
        n = result["n_slices"]
        self.slice_slider.config(to=n - 1)
        self.slice_var.set(n // 2)
        self.process_btn.config(state=tk.NORMAL)
        self._update_display()

    def _on_error(self, exc):
        self.process_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text=f"Erro: {exc}")

    # ── Renderização ─────────────────────────────────────────────────

    def _update_display(self, *_):
        if self.result is None:
            return

        idx       = self.slice_var.get()
        n         = self.result["n_slices"]
        opacity   = self.opacity_var.get()

        orig_slice  = self.result["volume"][idx]
        seg_slice   = self.result["seg_map"][idx]
        anom_slice  = self.result["anomaly_mask"][idx]

        self.slice_lbl.config(text=f"{idx + 1} / {n}")

        # Painel esquerdo — original
        self.ax_orig.clear()
        self.ax_orig.imshow(orig_slice, cmap="gray", interpolation="bilinear")
        self.ax_orig.set_title("Original", color="white", fontsize=10)
        self.ax_orig.axis("off")

        # Painel direito — segmentação + anomalias
        self.ax_seg.clear()
        self.ax_seg.imshow(orig_slice, cmap="gray", interpolation="bilinear")

        if self.show_seg_var.get():
            h, w = seg_slice.shape
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            for lbl, rgba in enumerate(TISSUE_COLORS):
                if lbl == 0:
                    continue
                mask = seg_slice == lbl
                overlay[mask, 0] = rgba[0]
                overlay[mask, 1] = rgba[1]
                overlay[mask, 2] = rgba[2]
                overlay[mask, 3] = rgba[3] * opacity
            self.ax_seg.imshow(overlay, interpolation="bilinear")

        if self.show_anom_var.get() and anom_slice.any():
            h, w = anom_slice.shape
            anom_overlay = np.zeros((h, w, 4), dtype=np.float32)
            anom_overlay[anom_slice, 0] = ANOMALY_RGBA[0]
            anom_overlay[anom_slice, 1] = ANOMALY_RGBA[1]
            anom_overlay[anom_slice, 2] = ANOMALY_RGBA[2]
            anom_overlay[anom_slice, 3] = ANOMALY_RGBA[3] * opacity
            self.ax_seg.imshow(anom_overlay, interpolation="bilinear")

        self.ax_seg.set_title("Segmentação + Lesões", color="white", fontsize=10)
        self.ax_seg.axis("off")

        self.canvas.draw_idle()
        self._update_info(idx)

    def _update_info(self, idx):
        if self.result is None:
            return

        seg        = self.result["seg_map"][idx]
        anom       = self.result["anomaly_mask"][idx]
        brain_2d   = self.result["brain_mask"][idx]
        total      = int(brain_2d.sum())

        lines = [f"Slice {idx + 1}"]
        for lbl, name in enumerate(TISSUE_NAMES[1:], start=1):
            count = int((seg == lbl).sum())
            pct   = (count / total * 100) if total > 0 else 0.0
            lines.append(f"  {name}: {pct:.1f}%")
        lines.append(f"  Lesões: {int(anom.sum())} voxels")

        self.info_lbl.config(text="\n".join(lines))

    def _step_slice(self, delta):
        if self.result is None:
            return
        n       = self.result["n_slices"]
        new_val = max(0, min(n - 1, self.slice_var.get() + delta))
        self.slice_var.set(new_val)
        self._update_display()


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    SegmentationViewer(root)
    root.mainloop()
