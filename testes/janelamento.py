import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the DICOM file
dicom_path = 'your_mri_image.dcm'
ds = pydicom.dcmread(dicom_path)

# Extract the raw pixel array
pixel_array = ds.pixel_array

# 2. Extract Window and Level metadata
# Pydicom will fetch these automatically if they are embedded in the DICOM header
window_center = ds.get('WindowCenter', 40)
window_width = ds.get('WindowWidth', 80)

# 3. Calculate minimum and maximum visible display values
min_value = window_center - window_width / 2
max_value = window_center + window_width / 2

# 4. Display the windowed image
plt.figure(figsize=(8, 8))
plt.imshow(pixel_array, cmap='gray', vmin=min_value, vmax=max_value)
plt.title(f'Windowed DICOM (WW: {window_width}, WC: {window_center})')
plt.axis('off')
plt.show()
