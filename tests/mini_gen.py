import os
import torch
import torch.nn as nn

out_dir = os.path.join(os.path.dirname(__file__), "specific_operations")
os.makedirs(out_dir, exist_ok=True)


def dump(model, inputs, name):
    torch.onnx.export(
        model, inputs, os.path.join(out_dir, name),
        input_names=["input"], output_names=["output"], dynamo=False
    )
    print(name, "done")


class ReluNet(nn.Module):
    def forward(self, x):
        return x.relu()


class AddNet(nn.Module):
    def forward(self, a, b):
        return a + b


class MulNet(nn.Module):
    def forward(self, a, b):
        return a * b


class MatMulNet(nn.Module):
    def forward(self, a, b):
        return a @ b


class GemmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(x)


x1 = torch.randn(1, 3, 4, 4)
x2 = torch.randn(1, 3, 4, 4)

dump(ReluNet(), x1, "relu.onnx")
dump(AddNet(), (x1, x2), "add.onnx")
dump(MulNet(), (x1, x2), "mul.onnx")

a = torch.randn(1, 4, 3)
b = torch.randn(1, 3, 5)
dump(MatMulNet(), (a, b), "matmul.onnx")

dump(GemmNet(), torch.randn(1, 10), "gemm.onnx")
dump(ConvNet(), torch.randn(1, 3, 8, 8), "conv.onnx")