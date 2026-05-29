"""
Conversor DICOM com Janelamento
Lê todos os DICOMs de D:\...\MR-MS-new, aplica WC=1248 / WW=2300
e salva os novos DICOMs em MR-MS-new-janelado.
"""

import os
import glob
import shutil
import pydicom
import numpy as np

# ── Configurações ────────────────────────────────────────────────
SOURCE_ROOT = r"D:\Users\paulo\PycharmProjects\pythonProjectUnsupervisedSegmentationBrainTC\dataset\MR-MS-new"
DEST_ROOT   = r"D:\Users\paulo\PycharmProjects\pythonProjectUnsupervisedSegmentationBrainTC\dataset\MR-MS-new-janelado"

WINDOW_CENTER = 1248
WINDOW_WIDTH  = 2300
# ─────────────────────────────────────────────────────────────────


def apply_window_uint16(pixel_array: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Aplica janelamento e remapeia para uint16 (0–65535).
    Mantém profundidade de bits máxima para preservar compatibilidade DICOM.
    """
    low  = center - width / 2.0
    high = center + width / 2.0
    clipped = np.clip(pixel_array.astype(np.float32), low, high)
    scaled  = (clipped - low) / (high - low) * 65535.0
    return scaled.astype(np.uint16)


def convert_dicom(src_path: str, dst_path: str):
    ds = pydicom.dcmread(src_path)

    # Lê o pixel_array original (já aplica RescaleSlope/Intercept se existir)
    try:
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        print(f"    [AVISO] Não foi possível ler pixel_array de {src_path}: {e}")
        shutil.copy2(src_path, dst_path)
        return

    # Aplica janelamento
    windowed = apply_window_uint16(pixel_array, WINDOW_CENTER, WINDOW_WIDTH)

    # Atualiza os metadados de pixel
    ds.PixelData = windowed.tobytes()
    ds.BitsAllocated    = 16
    ds.BitsStored       = 16
    ds.HighBit          = 15
    ds.PixelRepresentation = 0          # unsigned

    # Atualiza os campos de janela no header
    ds.WindowCenter = WINDOW_CENTER
    ds.WindowWidth  = WINDOW_WIDTH

    # Remove RescaleSlope/Intercept para evitar dupla conversão
    for tag in ("RescaleSlope", "RescaleIntercept", "RescaleType"):
        if tag in ds:
            del ds[tag]

    # Força Transfer Syntax não-comprimida (Explicit VR Little Endian)
    # Necessário porque os DICOMs originais podem estar comprimidos (JPEG/JPEG2000)
    # e ao substituir o PixelData por bytes raw precisamos mudar a sintaxe.
    if hasattr(ds, "file_meta") and ds.file_meta is not None:
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    else:
        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID    = ds.get("SOPClassUID", "1.2.840.10008.5.1.4.1.1.4")
        file_meta.MediaStorageSOPInstanceUID = ds.get("SOPInstanceUID", pydicom.uid.generate_uid())
        file_meta.TransferSyntaxUID          = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta = file_meta

    ds.is_implicit_VR  = False
    ds.is_little_endian = True

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ds.save_as(dst_path, write_like_original=False)


def main():
    # Descobre todas as sub-pastas de pacientes
    patient_dirs = sorted([
        d for d in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, d))
    ])

    total_files   = 0
    total_patients = len(patient_dirs)

    print(f"Origem : {SOURCE_ROOT}")
    print(f"Destino: {DEST_ROOT}")
    print(f"WC={WINDOW_CENTER}  WW={WINDOW_WIDTH}")
    print(f"Pacientes encontrados: {total_patients}")
    print("=" * 60)

    for p_idx, patient in enumerate(patient_dirs, 1):
        src_patient_dir = os.path.join(SOURCE_ROOT, patient)
        dst_patient_dir = os.path.join(DEST_ROOT,   patient)

        dcm_files = sorted(
            glob.glob(os.path.join(src_patient_dir, "*.dcm")),
            key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
            if os.path.splitext(os.path.basename(f))[0].isdigit()
            else os.path.basename(f)
        )

        if not dcm_files:
            print(f"[{p_idx:02d}/{total_patients}] {patient}: nenhum .dcm encontrado, pulando.")
            continue

        print(f"[{p_idx:02d}/{total_patients}] {patient}: {len(dcm_files)} slices...", end=" ", flush=True)

        for f_idx, src_file in enumerate(dcm_files):
            filename   = os.path.basename(src_file)
            dst_file   = os.path.join(dst_patient_dir, filename)
            convert_dicom(src_file, dst_file)

            # Progresso simples inline
            pct = int((f_idx + 1) / len(dcm_files) * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"\r[{p_idx:02d}/{total_patients}] {patient}: [{bar}] {pct:3d}%  ({f_idx+1}/{len(dcm_files)})", end="", flush=True)

        total_files += len(dcm_files)
        print(f"\r[{p_idx:02d}/{total_patients}] {patient}: concluído ({len(dcm_files)} slices)      ")

    print("=" * 60)
    print(f"Conversão finalizada!")
    print(f"Total de arquivos convertidos: {total_files}")
    print(f"Pasta de saída: {DEST_ROOT}")


if __name__ == "__main__":
    main()
