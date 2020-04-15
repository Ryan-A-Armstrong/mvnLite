'''
Programable Filter for ParaView
Import exodus file with velocity values

If using a weighted average:
1. PointData to CellData
2. MeshQuality filer for area/volume
3. uV = Calculator u*Quality
4. CellData to PointData (averaging occurs here)
5. u_w = Calculator uV/Quality

Get Wall Shear Stress
1. Gradient of Unstructured Data (Use default 'Gradients' name)
2. Extract Surfaces
3. Generate Surface Normals (Use default 'Normals' name)
4. Programable filter with this as the script.

5. Update mu to correct viscosity if you want meaningful units.

Calculations based on: https://physics.stackexchange.com/questions/250227/wall-shear-stress
Final value is magnitude of the T_wss defined in the link
'''

import numpy as np

mu = 1
grad_u = inputs[0].PointData['Gradients']
norms = inputs[0].PointData['Normals']

grad_u_tensors = []
norms_list = []
wss = []

for grads in grad_u:
    T_u = np.array([[grads[0][0], grads[0][1], grads[0][2]], [grads[1][0], grads[1][1], grads[1][2]],
                    [grads[2][0], grads[2][1], grads[2][2]]])
    grad_u_tensors.append(T_u)

for n in norms:
    norms_list.append(np.array([max(n[0]), max(n[1]), max(n[2])]))

it = 0
for n in norms_list:
    T = mu * np.dot((grad_u_tensors[it] + grad_u_tensors[it].T), n)
    T_wss = T - np.dot(np.dot(T, n), n)
    T_wss = np.sqrt(max(T_wss[0]) ** 2 + max(T_wss[1]) ** 2 + max(T_wss[2]) ** 2)
    wss.append(T_wss)
    it += 1

wss = np.array(wss)
output.PointData.append(wss, 'T_wss')
print('Done')
