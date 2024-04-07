# README

## Introduction

This repository contains the implementation of a methodology for automated segmentation of brain structures in cranial tomography exams.
The main objective is to speed up the annotation process and minimize manual intervention through the use of segmentation techniques based on deep neural networks.
This approach is based on [4], Kim, K., Kanezaki, A., & Tanaka, Y. (2020).

## Methodology

The methodology adopted in this work consists of an approach that employs automated brain region segmentation techniques. The validation flow of the proposed methodology is illustrated in Figure~\ref{fig: flowchart_methodology}.

In the data pre-processing stage (Section \ref{Pre_Processing}), tomography exams are standardized and subjected to interpolation and windowing procedures to eliminate irrelevant information.

The architecture of the proposed segmenter is detailed in Section \ref{sec:segmentor}, where the procedures for feature extraction, data permutation, calculation of evaluation metrics, among others, are presented.

The optimization of the model's hyperparameters is discussed in Section \ref{optimizando_hiper}, where issues such as optimizer selection, number of convolutional filters, number of convolutional layers, among others, are addressed.

Finally, the evaluation of the proposed model is described in Section \ref{secao_avaliacao}, where the comparison metrics with other methodologies established in the literature and the qualitative evaluation by medical radiologists are presented.

## How to use

To use this code, you must follow the instructions contained in the documentation for each system component. Make sure you have all dependencies installed before running the code.

## References

[1] Brudfors, M., Balbastre, Y., Lindhe, O., & Maier, A. (2020). Flexible and robust convolutional neural networks for brain tumor segmentation with self-generated anatomical models. arXiv preprint arXiv:2008.06458.

[2] Amanatiadis, A., Adeshina, S. A., & Delopoulos, A. (2009). A survey of medical image registration—under review.

[3] Parsania, S., Razavi, S. N., & Pourghassem, H. (2014). Review of image interpolation methods in medical imaging applications.

[4] Kim, K., Kanezaki, A., & Tanaka, Y. (2020). Unsupervised brain tumor segmentation by generating reliable image-level predictions. arXiv preprint arXiv:2004.15021.

[5] Bertels, J., Al Barashdi, H., Subbaiah, P. V., Bovenkamp, J., & Veen, L. D. V. D. (2019). Optimizing convolutional neural networks for bone age assessment using transfer learning and data augmentation.

[6] Eelbode, T., Dhondt, B., Pizurica, A., & Philips, W. (2020). Optimization of the FCM clustering algorithm for fast and accurate segmentation of white matter lesions from MR images.

[7] Thada, V. R., & Kumari, S. (2013). Comparison of segmentation algorithms for brain MRI.

## Contributions

Contributions are welcome! Please feel free to open an issue or submit a pull request if you would like to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.