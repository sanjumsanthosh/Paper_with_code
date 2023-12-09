import torch
from Models import LeNet


torch.onnx.export(
    LeNet().eval(),
    torch.randn(1, 1, 28, 28),
    "lenet.onnx",
    input_names=["input"],
    output_names=["output"],
)