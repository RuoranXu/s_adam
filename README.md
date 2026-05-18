# S-Adam: Singularity-aware Adam

This repository contains experimental code for **Singularity-aware Adam (S-Adam)**, an optimizer for non-smooth deep learning objectives. S-Adam extends AdamW with a local geometric instability probe and an adaptive damping factor, aiming to reduce gradient chattering near non-smooth singularities such as ReLU kinks and quantization operators.

The accompanying manuscript is:

> **Singularity-aware Optimization via Randomized Geometric Probing: Towards Stable Non-smooth Optimization**

Paper sources and figures are kept under [`Camera_Ready/`](Camera_Ready/).

## Core Idea

Modern neural networks often violate the smooth-gradient assumption because they contain non-smooth components such as ReLU activations, fake quantization, and sparsity-inducing operations. Near these singularities, adaptive optimizers can accumulate conflicting gradient signals and oscillate.

S-Adam addresses this by:

- estimating **Local Geometric Instability (LGI)** through randomized directional loss probes;
- using LGI as a proxy for local non-smoothness / Clarke subdifferential spread;
- applying a damping factor `exp(-lambda_LGI * LGI)` to the AdamW update;
- behaving similarly to AdamW in smooth regions where LGI is small.

In code, each S-Adam optimizer step requires a closure because the optimizer evaluates the baseline loss and several perturbed forward passes to estimate LGI.

## Repository Layout

```text
.
|-- CIFAR10_resnet_rebuttal12.py
|-- CIFAR10_resnet_rebuttal12_ablationk.py
|-- CIFAR100_ResNet_rebuttal12.py
|-- CIFAR100_ResNet_rebuttal12_ablationk.py
|-- CIFAR100_QAT_rebutttal12.py
|-- CIFAR100_QAT_rebuttal12_ablationk.py
|-- ImageNet10_resnet_rebuttal12.py
|-- ImageNet10_resnet_rebuttal12_ablationk.py
|-- ImageNet10_QAT_rebuttal12.py
|-- ImageNet10_QAT_rebuttal12_ablationk.py
|-- ImageWoof_resnet_rebuttal12.py
|-- ImageWoof_resnet_rebuttal12_ablationk.py
|-- ImageWoof_QAT_rebuttal12.py
|-- ImageWoof_QAT_rebuttal12_ablationk.py
|-- TinyImageNet_QAT_rebuttal12.py
|-- TinyImageNet_QAT_rebuttal12_ablationk.py
`-- Camera_Ready/
```

The scripts are standalone experiment files. Each file includes the S-Adam implementation, baseline optimizers, data loading, training / evaluation loops, and plotting code.

## Environment

The experiments were prepared with the following environment:

```text
Python==3.10.19
torch==2.9.1
torchvision==0.24.1
numpy==2.2.6
matplotlib==3.10.8
Pillow==12.1.0
psutil==7.2.1
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install torch==2.9.1 torchvision==0.24.1 numpy==2.2.6 matplotlib==3.10.8 Pillow==12.1.0 psutil==7.2.1
```

For GPU runs, install the PyTorch wheel that matches your CUDA version if the default wheel is not suitable for your system.

## Data

Most scripts store datasets under `./data`.

| Dataset | Handling |
| --- | --- |
| CIFAR-10 | Downloaded automatically by `torchvision.datasets.CIFAR10`. |
| CIFAR-100 | Downloaded automatically by `torchvision.datasets.CIFAR100`. |
| ImageWoof2-160 | Downloaded automatically by the custom `ImageWoof` class when `download=True`. |
| TinyImageNet | Downloaded automatically by the custom `TinyImageNet` class when `download=True`. |
| ImageNet-10 | Requires a local ImageNet-style directory at `./data/imagenet`. |

ImageNet-10 expects:

```text
data/imagenet/
|-- train/
|   |-- n01440764/
|   |-- n02102040/
|   `-- ...
`-- val/
    |-- n01440764/
    |-- n02102040/
    `-- ...
```

The selected ImageNet-10 WordNet IDs are:

```text
n01440764, n02102040, n02979186, n03000684, n03028079,
n03394916, n03417042, n03425413, n03445777, n03888257
```

## Running Experiments

Run a script directly from the repository root:

```bash
python CIFAR100_QAT_rebutttal12.py
```

Other examples:

```bash
python CIFAR10_resnet_rebuttal12.py
python CIFAR100_ResNet_rebuttal12.py
python ImageWoof_resnet_rebuttal12.py
python TinyImageNet_QAT_rebuttal12.py
python ImageNet10_QAT_rebuttal12.py
```

The scripts automatically select CUDA when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Training logs and plots are written to the repository root, for example:

```text
training_log_*.txt
sadam_vs_*.png
ablationk_*.png
```

## Experiment Groups

| Script pattern | Purpose |
| --- | --- |
| `*_QAT_rebuttal12.py` | Quantization-aware training experiments with S-Adam and baselines. |
| `*_resnet_rebuttal12.py` | ResNet18 transfer-learning experiments with S-Adam and baselines. |
| `*_ablationk.py` | Ablation over the number of randomized LGI probe directions `k_directions`. |

The main baselines are:

- `AdamW`
- `ProxSGD`
- `RMSprop`
- `SAdam` (ours)

## Key Hyperparameters

Typical S-Adam settings used in the scripts:

```python
SAdam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2,
    k_directions=8,
    sigma=0.01,
    lgi_lambda=2.0,
)
```

Important parameters:

- `k_directions`: number of randomized directional probes for LGI estimation.
- `sigma`: perturbation radius for randomized probing.
- `lgi_lambda`: damping intensity for high-LGI regions.
- `stabilize_eps`: numerical stabilizer in the LGI ratio.

The ablation scripts vary `k_directions` to evaluate the cost / accuracy trade-off.

## Using S-Adam in Your Own Training Loop

S-Adam requires a closure with a `backward` flag:

```python
optimizer = SAdam(model.parameters(), lr=1e-3, k_directions=8, sigma=0.01, lgi_lambda=2.0)

def closure(backward=True):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    if backward:
        loss.backward()
    return loss

loss = optimizer.step(closure)
```

The optimizer uses `closure(backward=True)` for the real gradient update and `closure(backward=False)` for no-gradient forward probes.

## Notes for Reproducibility

- The scripts are configured as standalone experiments rather than command-line tools.
- To change batch size, epochs, learning rate, or S-Adam hyperparameters, edit the constants inside the relevant script.
- ResNet experiments use torchvision ResNet18 with ImageNet weights when available.
- S-Adam performs additional forward passes per update, so runtime depends strongly on `k_directions`.
- CUDA memory is reported through `torch.cuda.max_memory_allocated()` when CUDA is available.

## Citation

If you use this code, please cite the associated paper. Add the final BibTeX entry here after publication.

```bibtex
@article{sadam2026,
  title   = {Singularity-aware Optimization via Randomized Geometric Probing: Towards Stable Non-smooth Optimization},
  author  = {Anonymous},
  journal = {Manuscript},
  year    = {2026}
}
```

## License

No license has been specified yet. Add a `LICENSE` file before public release if you intend others to reuse or redistribute the code.
