
import time
import nnet
import cubelattice as cl
import multiprocessing
from functools import partial
from scipy.linalg import null_space
import numpy as np

def check_sampling(vfl, samples):
    lcs = []
    for n in range(vfl.lattice.shape[1]):
        afacet = vfl.vertices[vfl.lattice[:,n],:]
        other_vs = vfl.vertices[~vfl.lattice[:,n],:]
        ones = np.ones((afacet.shape[0],1))
        afacet_ones = np.concatenate((afacet, ones), axis=1)
        lc = null_space(afacet_ones)  # linear constraint

        val = np.sum(np.dot(other_vs, lc[:5]) + lc[5])
        if val > 0:
            lc = -lc

        if len(lcs) == 0:
            lcs = lc
        else:
            lcs = np.concatenate((lcs, lc), axis=1)

    vals = np.dot(samples, lcs[:5]) + lcs[5]
    num = len(np.nonzero(np.all(vals<0, axis=1))[0])
    return num

def single_unsafety(vfl, p, datax, outputs):

    matrix_A = p[1][0]
    vector_d = p[1][1]
    for n in range(len(matrix_A)):
        A = matrix_A[[n]]
        d = vector_d[[n]]

        vfl = vfl.single_split(A, d)
        if not vfl:
            print('here')
            return

    sampling_unsafe = check_sampling(vfl, datax)
    outputs.put([vfl.vertices[0],sampling_unsafe])
    return



# def backtrack(vfl_sets, p, datax, pool):
#
#     vfls_unsafe = vfl_sets
#     processes = []
#
#     outputs = Queue()
#     # for vfl in vfls_unsafe:
#     #     processes.append(Process(target=single_unsafety, args=(vfl, p, shared_samplings, inputs_unsafe)))
#     for vfl in vfls_unsafe:
#         processes.append(Process(target=single_unsafety, args=(vfl, p, datax, outputs)))
#
#     for ps in processes: ps.start()
#
#     unsafe_samplings = 0
#     unsafe_inputs = []
#     while not outputs.empty():
#         rls = outputs.get()
#         unsafe_inputs.append(rls[0])
#         unsafe_samplings += rls[1]
#
#     for ps in processes: ps.join()
#
#     while not outputs.empty():
#         rls = outputs.get()
#         unsafe_inputs.append(rls[0])
#         unsafe_samplings += rls[1]
#
#     # unsafe_samplings = 0
#     # unsafe_inputs = []
#     # for ps in processes:
#     #     rls = outputs.get()
#     #     unsafe_inputs.append(rls[0])
#     #     unsafe_samplings += rls[1]
#
#
#     unsafe_inputs = np.array(unsafe_inputs).T
#     return unsafe_inputs


def backtrack0(vfl_sets, pty, samples, pool):
    matrix_A = pty[1][0]
    vector_d = pty[1][1]

    vfls_unsafe = vfl_sets
    for n in range(len(matrix_A)):
        A = matrix_A[[n]]
        d = vector_d[[n]]
        temp = []
        for vfl in vfls_unsafe:
            subvfl0, subvfl1 = vfl.single_split(A, d)
            if subvfl0:
                temp.append(subvfl0)
        vfls_unsafe = temp

    if not vfls_unsafe:
        return torch.tensor([]), 0

    inputs_unsafe = []
    samples_unsafe = 0
    for vfl in vfls_unsafe:
        samples_unsafe += check_sampling(vfl, samples)
        inputs_unsafe.append(vfl.vertices[0])

    inputs_unsafe = np.array(inputs_unsafe).T
    return inputs_unsafe, samples_unsafe


def backtrack(vfl_sets, pty, samples, pool):
    matrix_A = pty[1][0]
    vector_d = pty[1][1]

    vfls_unsafe = vfl_sets
    vfls_safe = []
    for n in range(len(matrix_A)):
        A = matrix_A[[n]]
        d = vector_d[[n]]
        temp = []
        for vfl in vfls_unsafe:
            subvfl0, subvfl1 = vfl.single_split(A, d)
            if subvfl0:
                temp.append(subvfl0)
            else:
                vfls_safe.append(vfl)

        vfls_unsafe = temp

    # safe data
    inputs_safe = []
    outputs_safe = []
    for vfl in vfls_safe:
        inputs_safe.append(vfl.vertices[0])
        temp = np.dot(vfl.vertices[0], vfl.M.T)+vfl.b.T
        outputs_safe.append(temp[0])

    inputs_safe = np.array(inputs_safe)
    outputs_safe = np.array(outputs_safe)
    if torch.cuda.is_available():
        inputs_safe = torch.tensor(inputs_safe).cuda()
        outputs_safe = torch.tensor(outputs_safe).cuda()
    else:
        inputs_safe = torch.tensor(inputs_safe)
        outputs_safe = torch.tensor(outputs_safe)

    if not vfls_unsafe:
        data_unsafe = [torch.tensor([]), torch.tensor([])]
        data_safe = [inputs_safe, outputs_safe]
        return data_unsafe, data_safe, 0

    # unsafe data
    inputs_unsafe = []
    outputs_unsafe = []
    samples_unsafe = 0
    for vfl in vfls_unsafe:
        samples_unsafe += check_sampling(vfl, samples)
        inputs_unsafe.append(vfl.vertices[0])
        temp = np.dot(vfl.vertices[0], vfl.M.T)+vfl.b.T
        outputs_unsafe.append(temp[0])

    inputs_unsafe = np.array(inputs_unsafe)
    outputs_unsafe = np.array(outputs_unsafe)

    if torch.cuda.is_available():
        inputs_unsafe = torch.tensor(inputs_unsafe).cuda()
        outputs_unsafe = torch.tensor(outputs_unsafe).cuda()
    else:
        inputs_unsafe = torch.tensor(inputs_unsafe)
        outputs_unsafe = torch.tensor(outputs_unsafe)

    data_unsafe = [inputs_unsafe,outputs_unsafe]
    data_safe = [inputs_safe,outputs_safe]

    return data_unsafe, data_safe, samples_unsafe


def compute_unsafety(model, datax, pool):
    W = []
    b = []
    for name, param in model.named_parameters():
        if name[-4:] == 'ight':
            if torch.cuda.is_available():
                W.append(param.data.cpu().numpy())
            else:
                W.append(param.data.numpy())
        if name[-4:] == 'bias':
            if torch.cuda.is_available():
                temp = np.expand_dims(param.data.cpu().numpy(), axis=1)
                b.append(temp)
            else:
                temp = np.expand_dims(param.data.numpy(), axis=1)
                b.append(temp)

    for n, pty in enumerate(datax.properties):
        print('  Working on Property: ', n)
        lb = pty[0][0]
        ub = pty[0][1]
        nnet0 = nnet.nnetwork(W, b)
        initial_input = cl.cubelattice(lb, ub).to_lattice()

        outputSets = []
        nputSets0 = nnet0.singleLayerOutput(initial_input, 0)
        nputSets = []
        for apoly in nputSets0:
            nputSets.extend(nnet0.singleLayerOutput(apoly, 1))

        outputSets.extend(pool.imap(partial(nnet0.layerOutput, m=2), nputSets))
        outputSets = [item for sublist in outputSets for item in sublist]

        samples = datax.samples[n]

        # inputs_unsafe, samples_unsafe = backtrack0(outputSets, pty, samples, pool)
        # if len(inputs_unsafe)==0:
        #     print('  Result: Safe')
        # else:
        #     print('  Result: Unsafe')
        #     print('  Unsafe input volume: ', samples_unsafe/ len(samples))

        data_unsafe, data_safe, samples_unsafe = backtrack(outputSets, pty, samples, pool)
        outputs_unsafe = data_unsafe[1]
        if len(outputs_unsafe)!=0:
            sorted, indices = torch.sort(outputs_unsafe, dim=1, descending=False)
            indices = indices.cpu().numpy()
            xx = 1



    print('\n')
