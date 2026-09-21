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

# ---------- bigger graph built node by node ----------
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def build_big_graph(path):
    rng = np.random.default_rng(0)
    nodes = []
    initializers = []

    def weight(name, shape):
        # from_array writes data into raw_data, which the importer expects
        arr = rng.standard_normal(shape).astype(np.float32)
        initializers.append(numpy_helper.from_array(arr, name))
        return name

    def node(op_type, inputs, output, **attrs):
        nodes.append(helper.make_node(op_type, inputs, [output], name=output, **attrs))
        return output

    def conv(x, out, w_shape, bias=True, **attrs):
        inputs = [x, weight(out + ".w", w_shape)]
        if bias:
            inputs.append(weight(out + ".b", [w_shape[0]]))
        return node("Conv", inputs, out, kernel_shape=w_shape[2:], **attrs)

    # ---- conv branch: image [1, 3, 16, 16] ----
    x = conv("image", "conv1", [8, 3, 3, 3], pads=[1, 1, 1, 1])            # [1, 8, 16, 16]
    x = node("Relu", [x], "relu1")

    for i in (2, 3):
        skip = x
        y = conv(x, f"conv{i}", [8, 8, 3, 3], pads=[1, 1, 1, 1])          # [1, 8, 16, 16]
        y = node("Relu", [y], f"relu{i}")
        y = node("Add", [y, skip], f"res{i}")
        x = node("Mul", [y, weight(f"scale{i}", [1, 8, 1, 1])], f"scaled{i}")

    x = conv(x, "conv_down", [16, 8, 3, 3], pads=[1, 1, 1, 1], strides=[2, 2])  # [1, 16, 8, 8]
    x = node("Relu", [x], "relu_down")

    skip = x
    y = conv(x, "conv_dil", [16, 4, 3, 3], pads=[2, 2, 2, 2],
             dilations=[2, 2], group=4)                                     # [1, 16, 8, 8]
    y = node("Relu", [y], "relu_dil")
    x = node("Add", [y, skip], "res_dil")

    conv_out = conv(x, "conv_proj", [4, 16, 1, 1], bias=False)             # [1, 4, 8, 8]

    # ---- dense branch: context [8, 16] ----
    h1 = node("Gemm", ["context", weight("fc1.w", [32, 16]), weight("fc1.b", [32])],
              "fc1", transB=1)                                              # [8, 32]
    h1 = node("Relu", [h1], "fc1_relu")
    h2 = node("Gemm", [h1, weight("fc2.w", [32, 32]), weight("fc2.b", [32])],
              "fc2", alpha=0.5)                                             # [8, 32]
    h2 = node("Relu", [h2], "fc2_relu")
    h = node("Add", [h2, h1], "fc_res")
    m = node("MatMul", [h, weight("proj", [32, 8])], "proj_mm")            # [8, 8]
    m = node("Mul", [m, weight("gate", [8])], "gated")                     # [8, 8] * [8]
    ctx = node("Relu", [m], "ctx")

    # ---- merge ----
    y = node("MatMul", [conv_out, ctx], "merge_mm")                        # [1, 4, 8, 8] @ [8, 8]
    y = node("Add", [y, weight("merge_bias", [4, 1, 1])], "merge_add")
    nodes.append(helper.make_node("Relu", [y], ["out"], name="out_relu"))

    # ---- second output ----
    nodes.append(helper.make_node(
        "Gemm", [ctx, weight("head.w", [8, 4]), weight("head.b", [])], ["context_out"],
        name="head", transA=1))                                             # [8, 4], scalar C

    graph = helper.make_graph(
        nodes,
        "big_graph",
        inputs=[
            helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 16, 16]),
            helper.make_tensor_value_info("context", TensorProto.FLOAT, [8, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4, 8, 8]),
            helper.make_tensor_value_info("context_out", TensorProto.FLOAT, [8, 4]),
        ],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # full_check also runs ONNX's own shape inference, so wrong shapes fail here
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)
    print(os.path.basename(path), f"done ({len(nodes)} nodes)")


build_big_graph(os.path.join(out_dir, "big_graph.onnx"))