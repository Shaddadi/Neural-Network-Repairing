import torch
import numpy as np

def get_safe_inputs(model, unsafe_data, datax):
    model.eval()
    safe_ys = []
    safe_xs = []
    for n, aset in enumerate(unsafe_data):
        if len(aset[0]) == 0:
            continue

        X_unsafe = aset[0]
        with torch.no_grad():
            Y_new = model(X_unsafe)
        p = datax.properties[n]
        M, vec = torch.tensor(p[1][0]), torch.tensor(p[1][1])
        if torch.cuda.is_available():
            M, vec = M.cuda(), vec.cuda()

        res = torch.matmul(M, Y_new.T) + vec
        indx = torch.any(res>0,dim=0)
        safe_xs.append(X_unsafe[indx])
        safe_ys.append(Y_new[indx])

    safe_xs = torch.cat(safe_xs, dim=0)
    safe_ys = torch.cat(safe_ys, dim=0)

    return safe_xs, safe_ys


def correct_inputs(unsafe_data, datax, beta=1.0):
    corrected_ys = []
    org_Xs = []
    for n, aset in enumerate(unsafe_data):
        if len(aset[0]) == 0:
            continue

        X_unsafe = aset[0]
        Y_unsafe = aset[1]
        p = datax.properties[n]
        M, vec = torch.tensor(p[1][0]), torch.tensor(p[1][1])
        if torch.cuda.is_available():
            M, vec = M.cuda(), vec.cuda()

        for i in range(len(Y_unsafe)):
            unsafe_y = Y_unsafe[[i]]
            res = torch.matmul(M, unsafe_y.T) + vec
            if torch.any(res>0):
                continue

            min_indx = torch.argmax(res)

            delta_y = M[min_indx] * (-res[min_indx,0] + beta)/(torch.matmul(M[[min_indx]],M[[min_indx]].T))
            safe_y = unsafe_y + delta_y
            corrected_ys.append(safe_y)
            org_Xs.append(X_unsafe[[i]])

    corrected_ys = torch.cat(corrected_ys, dim=0)
    org_Xs = torch.cat(org_Xs, dim=0)

    return org_Xs, corrected_ys



# def correct_inputs(unsafe_inputs, datax):
#     corrected_ys = []
#     org_xs = []
#     for n, aset in enumerate(unsafe_inputs):
#         if len(aset) == 0:
#             continue
#
#         org_xs.append(aset)
#         p = datax.properties[n]
#         M, vec = torch.tensor(p[1][0]), torch.tensor(p[1][1])
#         if torch.cuda.is_available():
#             M, vec = M.cuda(), vec.cuda()
#
#         for i in range(len(aset)):
#             unsafe_x = aset[[i]]
#             dis = torch.sum(abs(datax.train_x-unsafe_x),dim=1)
#             sorted_index = torch.argsort(dis)
#             for j in range(len(sorted_index)//2):
#                 x0 = datax.train_x[sorted_index[2*j]]
#                 y0 = datax.train_y[sorted_index[[2*j]]]
#                 x1 = datax.train_x[sorted_index[2*j+1]]
#                 y1 = datax.train_y[sorted_index[[2*j+1]]]
#
#                 corrected_y = (torch.sum(unsafe_x-x0)/torch.sum(x1-x0))*(y1-y0) + y0
#                 res = torch.matmul(M, corrected_y.T) + vec
#                 if torch.any(res>0):
#                     corrected_ys.append(corrected_y)
#                     break
#
#     corrected_ys = torch.cat(corrected_ys, dim=0)
#
#     org_xs = torch.cat(org_xs,dim=0)
#     return org_xs, corrected_ys





