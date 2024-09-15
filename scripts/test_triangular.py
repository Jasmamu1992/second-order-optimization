import torch

DEBUG=False

n = 10
a = torch.rand(n, 1)
A = a @ a.t()
if DEBUG:
    print(A)
cA = A * 2

lower_indices = torch.tril_indices(A.shape[0], A.shape[1])
upper_indices = torch.triu_indices(A.shape[0], A.shape[1])
U = A[upper_indices[0], upper_indices[1]]
U = U * 2
if DEBUG:
    print(U)
A[upper_indices[0], upper_indices[1]] = U
A[lower_indices[0], lower_indices[1]] = A.t()[lower_indices[0], lower_indices[1]]

if DEBUG:
    print(A)
    print(cA)
print('norm: ', (A-cA).norm())
