import torch
from Models import LeNet


torch.onnx.export(
    LeNet().eval(),
    torch.randn(1, 3, 512, 512),
    "lenet.onnx",
    input_names=["input"],
    output_names=["output"],
)