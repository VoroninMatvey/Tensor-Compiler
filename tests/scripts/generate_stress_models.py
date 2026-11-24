"""Generate reproducible ONNX graphs and matching runner stdin. Run from the project root."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FAMILIES = ("conv", "mixed", "dense", "batched")


class Graph:
    def __init__(self, rng):
        self.rng = rng
        self.inputs = []
        self.nodes = []
        self.weights = []

    def input(self, name, shape):
        self.inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
        return name

    def weight(self, shape, scale):
        name = f"w{len(self.weights)}"
        value = self.rng.normal(0, scale, size=shape).astype(np.float32)
        self.weights.append(numpy_helper.from_array(value, name))
        return name

    def op(self, op_type, *inputs, **attrs):
        name = f"{op_type.lower()}_{len(self.nodes)}"
        self.nodes.append(helper.make_node(op_type, inputs, [name], name=name, **attrs))
        return name

    def finish(self, name, outputs):
        infos = [helper.make_tensor_value_info(key, TensorProto.FLOAT, shape)
                 for key, shape in outputs]
        graph = helper.make_graph(self.nodes, name, self.inputs, infos, self.weights)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(model, full_check=True)
        return model


def conv(g, x, in_channels, out_channels, group, kernel, dilation, stride):
    kh, kw = kernel
    dh, dw = dilation
    # Total padding preserves the input extent along every axis with stride 1.
    pad_h, pad_w = (kh - 1) * dh, (kw - 1) * dw
    top = int(g.rng.integers(pad_h + 1))
    left = int(g.rng.integers(pad_w + 1))
    channels_per_group = in_channels // group
    scale = 1 / math.sqrt(channels_per_group * kh * kw)
    inputs = [x, g.weight((out_channels, channels_per_group, kh, kw), scale)]
    if g.rng.integers(2):
        inputs.append(g.weight((out_channels,), 0.1))
    return g.op("Conv", *inputs, group=group, kernel_shape=kernel, dilations=dilation,
                strides=stride, pads=[top, left, pad_h - top, pad_w - left])


def image_graph(g, size, blocks, mixed, channels):
    h, w = size, size + int(g.rng.integers(1, 5))
    if channels is None:
        channels = int(g.rng.choice([4, 6, 8, 12]))
    groups = [count for count in sorted({1, 2, channels}) if channels % count == 0]
    sh, sw = (int(s) for s in g.rng.integers(1, 3, size=2))
    x = g.input("image", [1, 3, h, w])
    x = conv(g, x, 3, channels, 1, (3, 3), (1, 1), (sh, sw))
    h, w = (h + sh - 1) // sh, (w + sw - 1) // sw
    for _ in range(blocks):
        group = int(g.rng.choice(groups))
        kernel = tuple(int(d) for d in g.rng.integers(1, 4, size=2))
        dilation = tuple(int(d) for d in g.rng.integers(1, 3, size=2))
        branch = conv(g, x, channels, channels, group, kernel, dilation, (1, 1))
        if g.rng.integers(2):
            shortcut = conv(g, x, channels, channels, 1, (1, 1), (1, 1), (1, 1))
            branch = g.op("Add", branch, shortcut)
        branch = g.op("Relu", branch)
        gate = g.weight((channels, 1, 1), 0.5 / math.sqrt(blocks))
        x = g.op("Add", x, g.op("Mul", branch, gate))

    if not mixed:
        return [(x, [1, channels, h, w])]

    # A second input drives the dense branch; its result is both consumed and returned.
    k = int(g.rng.integers(3, 9))
    context = g.input("context", [w, k])
    context = g.op("Gemm", context, g.weight((k, k), 1 / math.sqrt(k)),
                   g.weight((k,), 0.1), transB=1, alpha=0.75, beta=0.5)
    context = g.op("Relu", context)
    result = g.op("MatMul", x, context)
    return [(result, [1, channels, h, k]), (context, [w, k])]


def matrix_graph(g, size, blocks, batched):
    m, k = size, size + int(g.rng.integers(1, 5))
    if batched:
        x = g.input("left", [2, 1, m, k])
        rhs = g.input("right", [1, 3, k, k])
        x = g.op("MatMul", x, rhs)
        shape = [2, 3, m, k]
    else:
        shape = [m, k]
        x = g.input("matrix", shape)

    for _ in range(blocks):
        hidden = int(g.rng.choice([k, k + 3, 2 * k]))
        if batched:
            weight = g.weight((1, 3, k, hidden), 1 / math.sqrt(k))
            branch = g.op("MatMul", x, weight)
        else:
            trans_b = int(g.rng.integers(2))
            weight_shape = (hidden, k) if trans_b else (k, hidden)
            weight = g.weight(weight_shape, 1 / math.sqrt(k))
            branch = g.op("Gemm", x, weight, g.weight((hidden,), 0.1), transB=trans_b)
        branch = g.op("Relu", branch)
        branch = g.op("MatMul", branch, g.weight((hidden, k), 1 / math.sqrt(hidden)))
        gate = g.weight((k,), 0.5 / math.sqrt(blocks))
        x = g.op("Add", x, g.op("Mul", branch, gate))
    return [(x, shape)]


def generate(case, family, seed, size, blocks, channels):
    model_seed, input_seed = np.random.SeedSequence(seed).spawn(2)
    g = Graph(np.random.default_rng(model_seed))
    if family in ("conv", "mixed"):
        outputs = image_graph(g, size, blocks, family == "mixed", channels)
    else:
        outputs = matrix_graph(g, size, blocks, batched=(family == "batched"))
    model = g.finish(case.name, outputs)
    onnx.save(model, case.with_suffix(".onnx"))
    rng = np.random.default_rng(input_seed)
    with case.with_suffix(".stdin.txt").open("w") as stream:
        for value in g.inputs:
            shape = tuple(d.dim_value for d in value.type.tensor_type.shape.dim)
            data = rng.uniform(-1, 1, size=shape).astype(np.float32)
            np.savetxt(stream, data.ravel(), fmt="%.9g")
    print(f"{case.name}: {len(g.nodes)} nodes, {len(outputs)} outputs", flush=True)


def run_case(case, build, compare_script):
    model = case.with_suffix(".onnx")
    library = case.with_suffix(".so").resolve()
    inputs = case.with_suffix(".inputs.bin")
    outputs = case.with_suffix(".outputs.txt")
    start = time.perf_counter()
    subprocess.run([build / "tensor_compiler", model, "--emit=so", "-O2", "-o", library],
                   check=True)
    compiled = time.perf_counter()
    with case.with_suffix(".stdin.txt").open() as stdin, outputs.open("w") as stdout:
        subprocess.run([build / "runner", library, model, inputs],
                       stdin=stdin, stdout=stdout, check=True)
    print(f"  compile {compiled - start:.3f}s, runner {time.perf_counter() - compiled:.3f}s",
          flush=True)
    subprocess.run([sys.executable, compare_script, model, inputs, outputs], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("temp/generated"))
    parser.add_argument("--family", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--seed", type=int, default=2026, help="Seed of the first model")
    parser.add_argument("--count", type=int, default=12, help="Total number of models")
    parser.add_argument("--blocks", type=int, default=3, help="Residual blocks per model")
    parser.add_argument("--size", type=int, default=12, help="Base spatial or matrix dimension")
    parser.add_argument("--channels", type=int, help="Feature channels for conv and mixed graphs")
    parser.add_argument("--run", action="store_true", help="Compile, run and compare every case")
    parser.add_argument("--build", type=Path, default=Path("build"), help="Directory with runner and tensor_compiler")
    args = parser.parse_args()
    if args.seed < 0 or min(args.count, args.blocks, args.size) < 1:
        parser.error("seed must be non-negative and count, blocks and size must be positive")
    if args.channels is not None and args.channels < 1:
        parser.error("channels must be positive")
    compare_script = Path(__file__).with_name("compare_runner_outputs.py")
    if args.run:
        for path in (args.build / "tensor_compiler", args.build / "runner", compare_script):
            if not path.is_file():
                parser.error(f"Missing file for --run: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for i in range(args.count):
        family = FAMILIES[i % len(FAMILIES)] if args.family == "all" else args.family
        seed = args.seed + i
        case = args.out / f"{family}_s{seed}_b{args.blocks}_n{args.size}"
        if family in ("conv", "mixed") and args.channels is not None:
            case = case.with_name(f"{case.name}_c{args.channels}")
        generate(case, family, seed, args.size, args.blocks, args.channels)
        if args.run:
            try:
                run_case(case, args.build.resolve(), compare_script)
            except (subprocess.CalledProcessError, OSError) as error:
                print(f"FAIL {case.name}: {error}", file=sys.stderr, flush=True)
                failed.append(case.name)
    if args.run:
        print(f"{args.count - len(failed)}/{args.count} cases passed", flush=True)
        if failed:
            print("Failed cases:", ", ".join(failed), file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
