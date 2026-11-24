"""Compare a runner input dump and its stdout against the original ONNX model."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


DTYPES = {
    onnx.TensorProto.FLOAT: np.dtype("float32"),
    onnx.TensorProto.INT64: np.dtype("int64"),
}


def tensor_spec(value):
    tensor = value.type.tensor_type
    if tensor.elem_type not in DTYPES:
        raise ValueError(f"{value.name}: expected a float32 or int64 tensor")
    if not tensor.HasField("shape") or any(
        not dim.HasField("dim_value") or dim.dim_value < 0 for dim in tensor.shape.dim
    ):
        raise ValueError(f"{value.name}: the runner requires a static shape")
    return DTYPES[tensor.elem_type], tuple(dim.dim_value for dim in tensor.shape.dim)


def read_inputs(graph, path):
    raw = path.read_bytes()
    initializers = {tensor.name for tensor in graph.initializer}
    inputs = {}
    offset = 0
    for value in graph.input:
        if value.name in initializers:
            raise ValueError(f"Runner input {value.name!r} overlaps with an initializer")
        dtype, shape = tensor_spec(value)
        count = math.prod(shape)
        end = offset + count * dtype.itemsize
        if end > len(raw):
            raise ValueError(f"{path}: missing bytes for input {value.name!r}")
        inputs[value.name] = np.frombuffer(raw, dtype, count, offset).reshape(shape)
        offset = end
    if offset != len(raw):
        raise ValueError(f"{path}: {len(raw) - offset} unexpected bytes after the last input")
    return inputs


def parse_values(line, dtype, count, name):
    tokens = line.split()
    if len(tokens) != count:
        raise ValueError(f"{name}: expected {count} values, got {len(tokens)}")
    try:
        with np.errstate(over="raise"):
            values = np.array(tokens, dtype=dtype)
        # A decimal such as 1e999 may silently parse as infinity even in float64.
        if dtype == np.float32:
            for i in np.flatnonzero(np.isinf(values)):
                if tokens[i].lower().lstrip("+-") not in ("inf", "infinity"):
                    raise ValueError(f"{tokens[i]!r} exceeds float32 range")
    except (ValueError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"{name}: invalid {dtype} value ({error})") from error
    return values


def compare_output(output, reference, line, rtol, atol):
    name = output.name
    dtype, shape = tensor_spec(output)
    if reference.shape != shape or reference.dtype != dtype:
        raise ValueError(f"{name}: ONNX declares {dtype} {shape}, "
                         f"reference returned {reference.dtype} {reference.shape}")
    expected = reference.ravel()
    actual = parse_values(line, dtype, expected.size, name)
    detail = ""
    if dtype == np.float32:
        # Compare in float64 so subtracting two finite float32 extremes cannot overflow.
        a = actual.astype(np.float64)
        b = expected.astype(np.float64)
        matches = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            max_error = np.abs(a[finite] - b[finite]).max()
            detail = f", max_abs_error(finite)={max_error:.6g}"
    else:
        matches = actual == expected

    if matches.all():
        print(f"OK {name} {shape}: {expected.size} values{detail}")
        return True

    i = int(np.flatnonzero(~matches)[0])
    index = tuple(int(d) for d in np.unravel_index(i, shape))
    print(f"FAIL {name} {shape}: {np.count_nonzero(~matches)}/{expected.size} differ, "
          f"first at {index}: runner={actual[i]}, reference={expected[i]}{detail}")
    return False


def compare(model_path, inputs_path, outputs_path, rtol, atol):
    model = onnx.load(model_path)
    outputs = model.graph.output
    if not outputs:
        raise ValueError("Model has no outputs")
    inputs = read_inputs(model.graph, inputs_path)
    reference = ReferenceEvaluator(model).run([out.name for out in outputs], inputs)

    passed = 0
    with outputs_path.open(encoding="utf-8") as stream:
        for output, expected in zip(outputs, reference, strict=True):
            name = stream.readline().rstrip("\n")
            if name != output.name:
                raise ValueError(f"Expected output {output.name!r}, got {name!r}")
            line = stream.readline()
            if line == "":
                raise ValueError(f"{name}: missing line of values")
            if compare_output(output, expected, line, rtol, atol):
                passed += 1
        if stream.readline() != "":
            raise ValueError(f"{outputs_path}: unexpected lines after the last output")
    print(f"{passed}/{len(outputs)} outputs matched (rtol={rtol:g}, atol={atol:g})")
    return passed == len(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("inputs", type=Path, help="Binary inputs saved by the runner")
    parser.add_argument("outputs", type=Path, help="Text redirected from runner stdout")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    if any(not math.isfinite(t) or t < 0 for t in (args.rtol, args.atol)):
        parser.error("Tolerances must be finite and non-negative")
    try:
        matched = compare(args.model, args.inputs, args.outputs, args.rtol, args.atol)
    except Exception as error:
        # ONNX loading and evaluation can raise different exception types per operator.
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
