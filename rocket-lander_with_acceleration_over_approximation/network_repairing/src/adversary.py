import torch
import numpy as np

def create_adversary(model, num=50000):
    lb_p2 = [0.7291269841269841, -0.4997465213085514, -0.4997465213085514, 0.3888888888888889, -0.5]
    ub_p2 = [0.8362698412698413, 0.4997465213085514, 0.4997465213085514, 0.5, -0.44]
    lb_p3 = [-0.13694444444444445, -0.00954929658551372, 0.4933803235848756, 0.36666666666666664, 0.36]
    ub_p3 = [-0.13158730158730159, 0.00954929658551372, 0.4997465213085514, 0.5, 0.5]
    lb_p4 = [-0.13694444444444445, -0.00954929658551372, 1.891353241777396e-20, 0.3888888888888889, 0.2]
    ub_p4 = [-0.13158730158730159, 0.00954929658551372, 1.5915494309189554e-05, 0.5, 0.3]

    # pps = [[lb_p2, ub_p2], [lb_p3, ub_p3], [lb_p4, ub_p4]]
    pps = [[lb_p2, ub_p2]]

    all_inputs = []
    all_outputs = []
    for p in range(len(pps)):
        lb, ub = pps[p][0], pps[p][1]
        input_samples = []
        for n in range(len(lb)):
            input_samples.append(np.random.uniform(lb[n], ub[n], num))

        input_samples = torch.tensor(input_samples).cuda().T
        all_inputs.append(input_samples)
        with torch.no_grad():
            output_samples = model(input_samples)

        new_outputs = unsafe_outputs(output_samples, p)
        all_outputs.append(new_outputs)

    all_inputs = torch.cat(all_inputs).cuda()
    all_outputs = torch.cat(all_outputs).cuda()

    return all_inputs, all_outputs


def unsafe_outputs(safe_outputs, pp):
    if pp == 0:
        safe_outputs[:,0] = torch.min(safe_outputs[:,1:5],dim=1)[0] - 0.001
    elif (pp == 1) or (pp == 2):
        safe_outputs[:, 0] = torch.max(safe_outputs[:, 1:5], dim=1)[0] + 0.001

    return safe_outputs







