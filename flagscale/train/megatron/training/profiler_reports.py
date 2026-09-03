"""Generate operator-to-kernel reports from PyTorch profiler events."""

import csv
import gzip
import json
from collections import defaultdict, deque
from pathlib import Path

DETAIL_FIELDS = [
    "operator_name", "kernel_name", "variant_index", "mapping_status",
    "input_shapes", "input_dtypes", "candidate_operators",
    "kernel_event_count", "kernel_time_us",
]
SUMMARY_FIELDS = [
    "operator_name", "kernel_name", "kernel_call_count", "kernel_time_us", "percent",
]
OPERATOR_FIELDS = ["operator_id", "operator_name", "operator_kind", "kernel_name"]

# These are backend/library identities, not generic tensor operation names. Keep
# them centralized so support for another communication backend is a data-only change.
COMMUNICATION_OPERATOR_MARKERS = (
    "record_param_comms", "torch.distributed", "distributed::", "_c10d", "c10d::",
    "symm_mem::", "custom_ar::", "processgroup", "alltoall_dispatch",
)
COMMUNICATION_KERNEL_MARKERS = (
    "nccl", "rccl", "hccl", "oneccl", "custom_all_reduce", "cross_device_reduce",
    "allreduce", "all_reduce", "allgather", "all_gather", "reducescatter",
    "reduce_scatter", "alltoall", "all_to_all", "sendrecv",
)
NON_COMPUTE_KERNEL_MARKERS = ("memcpy", "memset")
NON_COMPUTE_EVENT_NAMES = {
    "Lazy Function Loading",
    "Runtime Triggered Module Loading",
}
TORCH_COMPILE_PREFIXES = ("triton_", "CompiledFunction")


def _json_field(value):
    value = [] if value in (None, "") else value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _load_trace_metadata(trace_path):
    """Load CPU-op names and dtype occurrences from a Kineto Chrome trace."""
    if trace_path is None:
        return None, {}
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)

    cpu_operator_names = set()
    input_dtypes = defaultdict(deque)
    for event in trace.get("traceEvents", ()):
        if event.get("cat") != "cpu_op" or event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        cpu_operator_names.add(name)
        args = event.get("args", {})
        if "Input type" in args:
            key = (name, _json_field(args.get("Input Dims")))
            input_dtypes[key].append(args["Input type"])
    return cpu_operator_names, input_dtypes


def _is_communication(operator_name, kernel_name):
    operator = operator_name.lower()
    kernel = kernel_name.lower()
    return any(marker in operator for marker in COMMUNICATION_OPERATOR_MARKERS) or any(
        marker in kernel for marker in COMMUNICATION_KERNEL_MARKERS
    )


def _is_non_compute(operator_name, kernel_name, cpu_operator_names):
    if operator_name in NON_COMPUTE_EVENT_NAMES:
        return True
    # Category is more stable than CUDA/HIP API naming. If a trace is available,
    # reject events that Kineto did not classify as CPU operators.
    if cpu_operator_names is not None and operator_name not in cpu_operator_names:
        return True
    kernel = kernel_name.lower()
    return any(marker in kernel for marker in NON_COMPUTE_KERNEL_MARKERS)


def _operator_kind(name):
    if name == "null":
        return "unattributed"
    if name.startswith("aten::"):
        return "aten"
    if name.startswith(TORCH_COMPILE_PREFIXES):
        return "torch_compile"
    if "::" in name:
        return "custom"
    return "runtime_operator"


def _duration_us(kernel):
    duration = getattr(kernel, "duration", 0.0)
    return float(duration() if callable(duration) else duration)


def _time(value):
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _percent(value):
    return "<0.001%" if 0 < value < 0.001 else f"{value:.3f}%"


def build_kernel_report_rows(events, trace_path=None):
    """Build detailed and summary rows from ``torch.profiler`` events."""
    variants = defaultdict(lambda: [0, 0.0])
    summaries = defaultdict(lambda: [0, 0.0])
    cpu_operator_names, trace_dtypes = _load_trace_metadata(trace_path)

    for event in events:
        if getattr(event, "is_user_annotation", False):
            continue
        operator = str(getattr(event, "key", None) or getattr(event, "name", "null"))
        shapes = _json_field(
            getattr(event, "structured_input_shapes", None)
            or getattr(event, "input_shapes", None)
        )
        dtype_queue = trace_dtypes.get((operator, shapes))
        event_dtypes = getattr(event, "input_dtypes", None)
        if not event_dtypes and dtype_queue:
            event_dtypes = dtype_queue.popleft()
        kernels = getattr(event, "kernels", None) or ()
        if not kernels:
            continue
        dtypes = _json_field(event_dtypes)

        for kernel in kernels:
            name = str(getattr(kernel, "name", "null"))
            if _is_communication(operator, name) or _is_non_compute(
                operator, name, cpu_operator_names
            ):
                continue
            duration = _duration_us(kernel)
            variants[(operator, name, shapes, dtypes)][0] += 1
            variants[(operator, name, shapes, dtypes)][1] += duration
            summaries[(operator, name)][0] += 1
            summaries[(operator, name)][1] += duration

    grouped = defaultdict(list)
    for (operator, kernel, shapes, dtypes), (count, duration) in variants.items():
        grouped[(operator, kernel)].append((shapes, dtypes, count, duration))

    details = []
    for (operator, kernel), items in sorted(grouped.items()):
        items.sort(key=lambda item: (-item[3], item[0], item[1]))
        for index, (shapes, dtypes, count, duration) in enumerate(items, 1):
            details.append({
                "operator_name": operator,
                "kernel_name": kernel,
                "variant_index": index,
                "mapping_status": "operator_shape_matched",
                "input_shapes": shapes,
                "input_dtypes": dtypes,
                "candidate_operators": "null",
                "kernel_event_count": count,
                "kernel_time_us": _time(duration),
            })

    total = sum(item[1] for item in summaries.values())
    summary = []
    for (operator, kernel), (count, duration) in sorted(
        summaries.items(), key=lambda item: (-item[1][1], item[0])
    ):
        summary.append({
            "operator_name": operator,
            "kernel_name": kernel,
            "kernel_call_count": count,
            "kernel_time_us": _time(duration),
            "percent": _percent(duration / total * 100.0 if total else 0.0),
        })
    return details, summary


def build_operator_rows(summary_rows):
    """List unique operator-to-kernel mappings using stable grouped IDs."""
    pairs = {(row["operator_name"], row["kernel_name"]) for row in summary_rows}
    kind_order = {
        "aten": 0, "custom": 1, "runtime_operator": 2,
        "torch_compile": 3, "unattributed": 4,
    }
    pairs = sorted(
        pairs, key=lambda pair: (kind_order[_operator_kind(pair[0])], pair[0], pair[1])
    )
    operator_ids = {}
    rows = []
    for operator, kernel in pairs:
        operator_ids.setdefault(operator, len(operator_ids) + 1)
        rows.append({
            "operator_id": operator_ids[operator],
            "operator_name": operator,
            "operator_kind": _operator_kind(operator),
            "kernel_name": kernel,
        })
    return rows


def _write(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_kernel_reports(events, profile_dir, rank, trace_path=None):
    """Write detailed, summary, and operator-to-kernel reports for one rank."""
    details, summary = build_kernel_report_rows(events, trace_path)
    operators = build_operator_rows(summary)
    profile_dir = Path(profile_dir)
    details_path = profile_dir / f"rank-{rank}_kernel_details_report.csv"
    summary_path = profile_dir / f"rank-{rank}_kernel_summary.csv"
    operator_path = profile_dir / f"rank-{rank}_operator_list.csv"
    _write(details_path, DETAIL_FIELDS, details)
    _write(summary_path, SUMMARY_FIELDS, summary)
    _write(operator_path, OPERATOR_FIELDS, operators)
    return details_path, summary_path, operator_path
