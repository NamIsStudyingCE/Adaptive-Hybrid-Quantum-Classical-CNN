import sys
sys.path.insert(0, '.')
import torch
from src.models import build_model, MODEL_REGISTRY
import pennylane

print('PennyLane:', pennylane.__version__, '| torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
NEW = ['proposed_noent', 'proposed_no_cnot', 'classical_concat',
       'proposed_noisy_short', 'proposed_noisy_expected']

for name in NEW:
    try:
        m, p = build_model(name, in_features=512, num_classes=4)
        x = torch.randn(8, 512)
        out = m(x)
        loss = out.sum()
        loss.backward()
        gnorm = sum(pp.grad.abs().sum().item() for pp in m.parameters() if pp.grad is not None)
        print(f'[OK] {name:<26} params={p:>5} out={list(out.shape)} grad_sum={gnorm:.4f}')
    except Exception as e:
        print(f'[FAIL] {name}: {type(e).__name__}: {str(e)[:200]}')
