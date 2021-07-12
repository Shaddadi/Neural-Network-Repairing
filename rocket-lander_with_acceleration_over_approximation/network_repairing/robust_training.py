
from __future__ import print_function
import argparse
import sys
sys.path.insert(0, 'src')
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from src.reach import compute_unsafety
from src.correction import correct_inputs, get_safe_inputs
from utils import test, train, load_nnet
from utils import DATA
import multiprocessing
import scipy.io as sio
import os
import numpy as np
import time
from adversary import create_adversary
import matplotlib.pyplot as plt

# Asymmetric loss function
def asymMSE(y_true, y_pred):
    lossFactor = 40.0
    numOut = 5
    d = y_true - y_pred
    maxes = torch.argmax(y_true, dim=1)
    maxes_onehot = torch.nn.functional.one_hot(maxes, numOut)
    others_onehot = maxes_onehot - 1
    d_opt = d * maxes_onehot
    d_sub = d * others_onehot
    a = lossFactor * (numOut - 1) * (torch.square(d_opt) + torch.abs(d_opt))
    b = torch.square(d_opt)
    c = lossFactor * (torch.square(d_sub) + torch.abs(d_sub))
    d = torch.square(d_sub)
    loss = torch.where(d_sub > 0, c, d) + torch.where(d_opt > 0, a, b)
    return torch.mean(loss)



def train_adversary(candidate, datax, savepath, epochs=200, batch_size=2**14, lr=0.0005):

    ori_train_x, ori_train_y = datax.train_x, datax.train_y
    optimizer = optim.Adam(candidate.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.5)
    cpus = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(cpus)

    # ins, outs = create_adversary(candidate)
    # train_x = torch.cat((ori_train_x, ins), dim=0)
    # train_y = torch.cat((ori_train_y, outs), dim=0)

    train_x = ori_train_x
    train_y = ori_train_y

    unsafe_data, safe_data, property_result = compute_unsafety(candidate, datax, pool)
    best_pp = np.max(np.array(property_result))
    # best_pp = 0.0001

    for epoch in range(1, epochs + 1):
        # if os.path.isfile(savepath+"/cas_nnet.pt"):
        #     break

        print('Epoch of training: ', epoch)

        accuracy = test(candidate.eval(), datax)
        unsafe_data, safe_data, property_result = compute_unsafety(candidate, datax, pool)
        if np.any(np.array(property_result)>=best_pp) and (accuracy >= 0.94):
            best_pp = np.max(np.array(property_result))
            test(candidate, datax)
            torch.save(candidate.state_dict(), savepath+"/cas_nnet.pt")

        training_dataset = TensorDataset(train_x.cpu(), train_y.cpu())
        train_loader = DataLoader(training_dataset, batch_size, shuffle=True, num_workers=15)
        print('  Start training...')
        train(candidate.train(), train_loader, asymMSE, optimizer)
        scheduler.step()
        print('  The training is done\n')


def plot_distribution(unsafe_outputs, index):
    ## test
    if len(unsafe_outputs) != 0:
        for nn in range(4):
            line_x = np.linspace(-0.5, 0.5, 100)
            line_y = line_x
            plt.figure()
            plt.plot(line_x, line_y, '-r')
            xs = unsafe_outputs[:, 0].cpu().numpy()
            ys = unsafe_outputs[:, nn + 1].cpu().numpy()
            plt.plot(xs, ys, 'b.')
            plt.xlabel('y0')
            plt.ylabel('y' + str(nn + 1))
            plt.title('y0 w.r.t. y' + str(nn + 1))
            plt.savefig('figures/'+str(index)+'y0 w.r.t. y' + str(nn + 1) + '.png')
            plt.close()
    ##

def main(candidate, datax, savepath, epochs=300, batch_size=2**14, lr=0.0005, steps=2):

    t0 = time.time()

    if not os.path.isdir(savepath+'/models'):
        os.mkdir(savepath+'/models')
    if not os.path.isdir(savepath+'/logs'):
        os.mkdir(savepath+'/logs')

    ori_train_x, ori_train_y = datax.train_x, datax.train_y

    # optimizer = optim.SGD(candidate.parameters(), lr=lr,  momentum=0.9)
    optimizer = optim.Adam(candidate.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=500, gamma=0.5)
    cpus = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool(cpus)
    all_test_accuracy = []
    all_reach_vfls = []
    all_unsafe_vfls = []
    all_property_result = []
    all_test_loss = []
    all_times = []

    unsafe_data, safe_data, property_result, vfls_all, vfls_unsafe\
        = compute_unsafety(candidate, datax, pool)
    if len(safe_data[0]) != 0:
        ori_train_x = torch.cat((ori_train_x, safe_data[0]), dim=0)
        ori_train_y = torch.cat((ori_train_y, safe_data[1]), dim=0)

    all_property_result.append(property_result)
    accuracy, mseloss = test(candidate.eval(), datax)
    all_test_accuracy.append(accuracy)
    all_test_loss.append(mseloss)

    for epoch in range(1, epochs + 1):
        print('Epoch of training: ', epoch)

        unsafe_data, safe_data, property_result, vfls_all, vfls_unsafe\
            = compute_unsafety(candidate, datax, pool)

        all_reach_vfls.append(vfls_all)
        all_unsafe_vfls.append(vfls_unsafe)
        all_property_result.append(property_result)
        accuracy, mseloss = test(candidate.eval(), datax)
        all_test_accuracy.append(accuracy)
        all_test_loss.append(mseloss)

        if np.all([len(aset[0])==0 for aset in unsafe_data]) and (accuracy>=0.94):
            print('\nThe accurate and safe candidate model is found !\n')
            test(candidate, datax)
            print('\n\n')
            torch.save(candidate.state_dict(), savepath+"/models/acasxu_epoch" + str(epoch) + "_safe.pt")
            sio.savemat(savepath+'/logs/all_test_accuracy.mat', {'all_test_accuracy': all_test_accuracy, 'all_test_loss':all_test_loss})
            sio.savemat(savepath + '/logs/reach_sets.mat', {'all_reach_vfls': all_reach_vfls, 'all_unsafe_vfls':all_unsafe_vfls})
            sio.savemat(savepath+'/logs/all_property_result.mat', {'all_property_result': all_property_result})
            break

        if epoch % steps != 1:
            unsafe_data = [[[]]]

        if not np.all([len(aset[0])==0 for aset in unsafe_data]):
            unsafe_xs, corrected_ys = correct_inputs(unsafe_data,datax)
            print('  Unsafe_inputs: ', len(unsafe_xs))
            train_x = torch.cat((ori_train_x, unsafe_xs), dim=0)
            train_y = torch.cat((ori_train_y, corrected_ys), dim=0)
            # if len(safe_data[0]) != 0:
            #     train_x = torch.cat((train_x, safe_data[0]), dim=0)
            #     train_y = torch.cat((train_y, safe_data[1]), dim=0)
            # train_x = torch.cat((ori_train_x, safe_data[0]), dim=0)
            # train_y = torch.cat((ori_train_y, safe_data[1]), dim=0)
            # if len(safe_data[0]) != 0:
            #     ori_train_x = torch.cat((ori_train_x, safe_data[0]), dim=0)
            #     ori_train_y = torch.cat((ori_train_y, safe_data[1]), dim=0)
            #
            # train_x = torch.cat((ori_train_x, unsafe_xs), dim=0)
            # train_y = torch.cat((ori_train_y, corrected_ys), dim=0)
        else:
            # train_x = torch.cat((ori_train_x, safe_data[0]), dim=0)
            # train_y = torch.cat((ori_train_y, safe_data[1]), dim=0)
            train_x = ori_train_x
            train_y = ori_train_y

        training_dataset = TensorDataset(train_x.cpu(), train_y.cpu())
        train_loader = DataLoader(training_dataset, batch_size, shuffle=True,num_workers=15)
        print('  Start training...')
        train(candidate.train(), train_loader, asymMSE, optimizer)
        scheduler.step()
        print('  The training is done\n')

        if not np.all([len(aset[0])==0 for aset in unsafe_data]):
            safe_xs, safe_ys = get_safe_inputs(candidate, unsafe_data, datax)
            ori_train_x = torch.cat((ori_train_x, safe_xs), dim=0)
            ori_train_y = torch.cat((ori_train_y, safe_ys), dim=0)

        all_times.append(time.time()-t0)

        sio.savemat(savepath+'/logs/all_test_accuracy.mat', {'all_test_accuracy': all_test_accuracy})
        sio.savemat(savepath + '/logs/reach_sets.mat',{'all_reach_vfls': all_reach_vfls, 'all_unsafe_vfls': all_unsafe_vfls})
        sio.savemat(savepath+'/logs/all_property_result.mat', {'all_property_result': all_property_result})
        sio.savemat(savepath + '/logs/all_times.mat', {'all_times': all_times})
        if epoch % 1 == 0:
            torch.save(candidate.state_dict(), savepath+"/models/acasxu_epoch"+str(epoch)+".pt")


    pool.close()


if __name__ == '__main__':

    # for pra in [0, 1, 2, 3, 4]:
    #     for tau in ['00', '01', '02', '05', '10', '15', '20', '40', '60']:

            pra = 1
            tau = '15'

            t0 = time.time()
            foldername = 'adversary_models_all'
            network_name = 'HCAS_rect_v6_pra'+str(pra)+'_tau'+tau+'_50HU'
            # if not os.path.isfile(foldername+'/'+network_name+'/'+'cas_nnet.pt'):
            #     continue

            print(foldername+'/Network_' + 'pra_' + str(pra) + '_tau_' + tau)
            savepath = 'results/'+foldername+'/'+network_name
            if not os.path.isdir('results'):
                os.mkdir('results')
            if not os.path.isdir('results/'+foldername):
                os.mkdir('results/'+foldername)
            if not os.path.isdir('results/'+foldername+'/'+network_name):
                os.mkdir('results/'+foldername+'/'+network_name)


            network_name = 'HCAS_rect_v6_pra'+str(pra)+'_tau'+tau+'_50HU'
            nn_path0 = foldername + '/' + network_name + '/cas_nnet.pt'
            candidate = load_nnet(nn_path0)

            # data
            datax = DATA([pra, tau])

            # savepath = 'adversary_models_all/'
            # if not os.path.isdir(savepath):
            #     os.mkdir(savepath)
            # savepath = savepath + network_name
            # if not os.path.isdir(savepath):
            #     os.mkdir(savepath)
            # train_adversary(candidate, datax, savepath)

            main(candidate, datax, savepath)



