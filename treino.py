'''
custom by Paulo Victor dos Santos
Original by Kim
'''


# para fazer a gpu funcionar no pytorch
# siga este tutorial https://pub.towardsai.net/installing-pytorch-with-cuda-support-on-windows-10-a38b1134535e
# e finalmente instale pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html



import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.init
from dicom_to_nifti import converter
from validarMetodologia import executar_metodologia
import cv2

import time
import matplotlib.pyplot as plt




use_cuda = torch.cuda.is_available()

print("is cuda avaliable: {}".format(use_cuda))

def treino(nChannel, nConv, lambda_rotulo, lr, maxIter, debug_progress, debug_interval, visualize, stepsize_con, stepsize_sim, minLabels):

    # args = parser.parse_args()


    folder_dcm = 'dataset/MR-MS-new/P0008/'
    nifti_file = 'result/P0008.nii.gz'

    #defined as brain windowing
    # window_center = 40
    # window_width = 80

    print("starting convertion")
    convertion_time = time.time()
    # vol, affine = converter(folder_dcm, nifti_file, window_center, window_width)
    vol, affine = converter(folder_dcm, nifti_file, apply_window=False)
    print("--- %s seconds convertion ---" % (time.time() - convertion_time))



    train_time = time.time()
    # define volume size
    Z = vol.shape[0]
    H = 512
    W = 512
    exame_train = vol.reshape(Z, 512, 512)



    # CNN model
    class MyNet(nn.Module):
        def __init__(self, input_dim):
            '''

            :param input_dim:
            '''
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(nChannel)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(nConv - 1):
                self.conv2.append(nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1))
                self.bn2.append(nn.BatchNorm2d(nChannel))
            self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0)
            self.bn3 = nn.BatchNorm2d(nChannel)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu(x)
            x = self.bn1(x)

            for i in range(nConv - 1):
                x = self.conv2[i](x)
                x = F.relu(x)
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x



    # data = torch.from_numpy(np.array([exame_train.astype('float32') / 255.]))
    data = torch.tensor([exame_train / 255.], dtype=torch.float32)
    data = data.reshape(Z, H, W)

    if use_cuda:
        data = data.cuda()
    data = Variable(data)



    # train
    # model = MyNet(data.size(1))
    print("creating model")
    model = MyNet(1)
    print(model)
    if use_cuda:
        model.cuda()
    model.train()

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average=True)
    loss_hpz = torch.nn.L1Loss(size_average=True)



    HPy_target = torch.zeros(512 - lambda_rotulo, 512, nChannel)
    HPz_target = torch.zeros(512, 512 - lambda_rotulo, nChannel)

    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()


    optimizer = optim.SGD(model.parameters(), lr=lr)
    # usa a qtd de filtros pra poder fazer a matriz de cores
    # use a qtd of filter to do a color matriz
    label_colours = np.random.randint(255, size=(nChannel, 3))



    stoped = False

    # iteração para finalizar o algoritmo em caso de não encontrar os rótulos
    # iterate to train
    print("starting training iteration")
    for batch_idx in range(maxIter):
        print("batch {} of {}".format(batch_idx, maxIter), flush=True)
        batch_start_time = time.time()
        loss_medio = 0
        if stoped:
            break
        for slice in range(Z):
            slice_start_time = time.time()
            if debug_progress and (slice == 0 or ((slice + 1) % debug_interval == 0) or (slice == Z - 1)):
                print("[debug] iniciando slice {}/{} no batch {}/{}".format(slice + 1, Z, batch_idx + 1, maxIter), flush=True)

            data1 = data[slice, :, :]  #exame na escala de cinza 255

            data1 = data1.reshape(1,1,512,512)
            if use_cuda:
                data1 = data1.cuda()
            data1 = Variable(data1)


            # forwarding
            optimizer.zero_grad()
            # extrator de características
            features = model(data1)[0]



            # plt.imshow(output1[0,:,:].data.cpu().numpy())
            # plt.show()
            # permutador 1
            permutado = features.permute(1, 2, 0).contiguous().view(-1, nChannel)


            # posicao = 0
            # rows, cols = 5, 5
            # plt.figure(figsize=(60, 40))
            # fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row')

            # for row in range(rows):
            #     for col in range(cols):
            #         # ax[row, col].text(0.5, 0.5,
            #         #                   str((row, col)),
            #         #                   color="green",
            #         #                   fontsize=18,
            #         #                   ha='center')
            #         ax[row, col].imshow(features[posicao, :, :].data.cpu().numpy())
            #         posicao = posicao + 1

            # plt.show()




            # plt.imshow(output.data.cpu().numpy())
            # plt.show()
            # remodelador1
            outputHP = permutado.reshape((data1.shape[2], data1.shape[3], nChannel))

            # início cálculo continuidade espacial
            # spacial continuity loss
            HPy = outputHP[lambda_rotulo:, :, :] - outputHP[0:-lambda_rotulo, :, :]
            HPz = outputHP[:, lambda_rotulo:, :] - outputHP[:, 0:-lambda_rotulo, :]

            # continuity loss definition
            lhpy = loss_hpy(HPy, HPy_target)
            lhpz = loss_hpz(HPz, HPz_target)


            ignore, target = torch.max(permutado, 1)
            im_target = target.data.cpu().numpy()
            data_show = target.data.cpu().numpy()
            data_show = data_show.reshape(512, 512).astype(np.uint8)

            nLabels = len(np.unique(im_target))


            if visualize:
                im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
                im_target_rgb = im_target_rgb.reshape(512,512,3).astype(np.uint8)

                im_target_rgb = cv2.resize(im_target_rgb, (600, 600))
                data2 = cv2.resize(data_show, (600, 600))
                cv2.imshow("output", im_target_rgb)
                cv2.imshow("original", data2)
                cv2.waitKey(10)

            loss = stepsize_sim * loss_fn(permutado, target) + stepsize_con * (lhpy + lhpz)

            loss.backward()
            optimizer.step()
            loss_medio += loss.item()

            torch.save(model.state_dict(), 'result/model.pth')
            torch.save(optimizer.state_dict(), 'result/optimizer.pth')

            print(batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

            if debug_progress and (slice == 0 or ((slice + 1) % debug_interval == 0) or (slice == Z - 1)):
                print(
                    "[debug] finalizou slice {}/{} | t_slice={:.2f}s | t_batch={:.2f}s | labels={} | loss={:.6f}".format(
                        slice + 1,
                        Z,
                        time.time() - slice_start_time,
                        time.time() - batch_start_time,
                        nLabels,
                        loss.item()
                    ),
                    flush=True
                )

            if nLabels <= minLabels:
                print("loss_final :", loss.item(), " loss_medio :", loss_medio / Z, " nLabels :", nLabels, " reached minLabels :", minLabels, ".")
                stoped = True
                # break
                # return loss.item()
                return nLabels

            # return loss.item() comentado para evitar voltar sem terminar a otimização


    print("--- %s seconds trains ---" % (time.time() - train_time))



    # INICIO VALIDAÇÃO
    # exames_validar = {
    # "CQ500CT42": "dataset/validation/CQ500CT42/Unknown Study/CT PRE CONTRAST THIN/",
    # "CQ500CT195": "dataset/validation/CQ500CT195/Unknown Study/CT PRE CONTRAST THIN/",
    # "CQ500CT200": "dataset/validation/CQ500CT200/Unknown Study/CT Thin Plain/",
    # "CQ500CT299": "dataset/validation/CQ500CT299/Unknown Study/CT Thin Plain/",
    # "CQ500CT418": "dataset/validation/CQ500CT418/Unknown Study/CT Thin Plain/"
    # }


    exames_validar = {
    "P0009": "dataset/MR-MS-new/P0009/",
    "P0010": "dataset/MR-MS-new/P0010/",
    "P0011": "dataset/MR-MS-new/P0011/"
    }

    # D:\Users\paulo\PycharmProjects\pythonProjectUnsupervisedSegmentationBrainTC\dataset\MR-MS-new\P0008

    # for key in exames_validar:
    #     exam_time = time.time()
    #     exame_validar_local = exames_validar[key]
    #     loss_medio = executar_metodologia(use_cuda, model, label_colours, args, key, exame_validar_local, window_center, window_width)
    #     print("--- %s seconds to exam %s ---" % (time.time() - exam_time, key))
    #     print("finalizou o exame {} com loss médio de {}".format(key, loss_medio))
