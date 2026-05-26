"""
  Module: mr_window_filter.py
  Aplica filtro de janelamento (windowing) em imagens DICOM de ressonância magnética.
  Usa window_width/window_center (window/level) fornecidos pelo usuário.
"""

import numpy as np
from dicom_to_nifti import find_dicom_files, load_dicom_series, dicom_to_volume, convert_coords, write_nifti
import sys


class MRWindowFilter:
    """
    Aplica janelamento (windowing) em séries DICOM de ressonância magnética.

    Parâmetros
    ----------
    window_width : float
        Largura da janela (window). Ex: 1.929
    window_center : float
        Centro da janela (level). Ex: 1.261
    """

    def __init__(self, window_width: float, window_center: float):
        self.window_width = window_width
        self.window_center = window_center

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica o janelamento ao array de voxels.

        Valores abaixo do mínimo são truncados para o mínimo;
        valores acima do máximo são truncados para o máximo.
        O resultado é normalizado para o intervalo [0, 1].
        """
        img_min = self.window_center - self.window_width / 2.0
        img_max = self.window_center + self.window_width / 2.0

        windowed = image.copy()
        windowed = np.clip(windowed, img_min, img_max)

        # Normaliza para [0, 1]
        windowed = (windowed - img_min) / (img_max - img_min)

        return windowed.astype(np.float32)

    def converter(self, input_dicoms: str, output_nifti: str):
        """Carrega DICOMs, aplica janelamento e salva como NIFTI.

        Parâmetros
        ----------
        input_dicoms : str
            Caminho para a pasta com arquivos DICOM.
            Ex: 'dataset/MR-MS-new/P0008'
        output_nifti : str
            Caminho de saída do arquivo NIFTI.
            Ex: 'result/P0008_windowed.nii.gz'

        Retorna
        -------
        tuple (vol, affine)
            vol    : np.ndarray com o volume após janelamento (float32, normalizado [0,1])
            affine : matriz affine 4x4 do volume
        """
        files = find_dicom_files(input_dicoms)
        if not files:
            sys.stderr.write("Nenhum arquivo DICOM encontrado em: %s\n" % input_dicoms)
            return None, None

        series = load_dicom_series(files)
        if not series:
            sys.stderr.write("Não foi possível ler os arquivos DICOM.\n")
            return None, None

        vol, pixdim, mat = dicom_to_volume(series)

        convert_coords(vol, mat)

        vol = self.apply(vol)

        write_nifti(output_nifti, vol, mat)

        return vol, mat
