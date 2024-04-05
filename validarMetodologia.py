'''
criado em 19/07/2023 para facilitar a execução da validação da metodologia
by Paulo Victor dos Santos
'''


import os
from torch.autograd import Variable
import cv2
import numpy as np
import torch.nn.init
import nibabel as nb
from dicom_to_nifti import converter


# continuity loss definition
loss_hpy = torch.nn.L1Loss(size_average=True)
loss_hpz = torch.nn.L1Loss(size_average=True)

# similarity loss definition
loss_fn = torch.nn.CrossEntropyLoss()

def executar_metodologia(use_cuda, model, label_colours, args, key, folder_dcm, window_center, window_width):
    try:
        loss_medio = 0
        HPy_target = torch.zeros(512 - 1, 512, args.nChannel)
        HPz_target = torch.zeros(512, 512 - 1, args.nChannel)

        if use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()


        nifti_file = "result\\{}_janelado.nii.gz".format(key)

        exame_teste, affine = converter(folder_dcm, nifti_file, window_center, window_width)
        empty_header = nb.Nifti1Header()
        empty_header.get_data_shape()

        Z = exame_teste.shape[0]

        fileName = "{}_{}.nii.gz".format(key,str(args.minLabels))
        img = nb.Nifti1Image(exame_teste.T, affine, empty_header)
        nb.save(img, os.path.join('build', fileName))

        # files_dcm = [os.path.join(os.getcwd(), folder_dcm, x) for x in os.listdir(folder_dcm)]
        # exame = np.array([read_dicom_file(path) for path in files_dcm])
        # exame_teste = np.array([read_dicom_file(folder_dcm)])
        exame1_teste = exame_teste.reshape(Z, 512, 512)
        nifti_teste = np.ones((Z, 512, 512), dtype=np.uint8)  # dummy data in numpy matrix



        for slice in range(Z):
            data1 = exame1_teste[slice, :, :]
            data_teste = torch.from_numpy(data1.reshape(1, 1, 512, 512).astype('float32'))
            if use_cuda:
                data_teste = data_teste.cuda()
            data_teste = Variable(data_teste)
            output_teste = model(data_teste)[0]
            output = output_teste.permute(1, 2, 0).contiguous().view(-1, args.nChannel)


            # CALCULANDO A LOSS
            outputHP = output.reshape((data1.shape[0], data1.shape[1], args.nChannel))
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]

            # continuity loss definition
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)

            # FIM CALCULANDO A LOSS

            ignore, target = torch.max(output, 1)
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))

            im_target_rgb = np.array([label_colours[c % args.nChannel] for c in im_target])
            im_target_rgb = im_target_rgb.reshape(512, 512, 3).astype(np.uint8)

            nifti_teste[slice, :, :] = im_target.reshape(512, 512).astype(np.uint8)

            im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
            data2 = cv2.resize(data1, (600, 600))
            cv2.imshow("output", im_target_rgb)
            cv2.imshow("original", data2)
            cv2.waitKey(10)
            loss = args.stepsize_sim * loss_fn(output, target) + args.stepsize_con * (lhpy + lhpz)
            loss.backward()
            loss_medio += loss.item()
            print('exame :', '/', key, '|', ' slice :', '/', slice, '|', ' rotulos :', nLabels, ' | loss_slice :', loss.item())

        # fileName = 'result_CQ500CT195_' + str(args.minLabels) + '.nii.gz'
        fileName = "result_{}_{}.nii.gz".format(key, str(args.minLabels))
        img = nb.Nifti1Image(nifti_teste.T, affine, empty_header)  # Save axis for data (just identity)
        img.header.get_xyzt_units()
        img.to_filename(os.path.join('build', fileName))  # Save as NiBabel file
        retorno = loss_medio / Z
        loss_medio = None
        return retorno
    except Exception as exc:
        print(exc)

        pass