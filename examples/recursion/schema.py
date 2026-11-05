"""Input/output format of EXPR-128, for the decision-pipe compiler (no task rules)."""
from task import FAULT, IN_ADDR, IN_ARG, IN_IMM, IN_NODE, IN_OP, IN_QUERY, IN_ROOT, IN_X, IN_Y, IN_Z, OPS, TIMEOUT

ENUMS = {"Op": OPS}
RECORDS = {
    "program": {"key": "addr", "indicator": IN_NODE,
                "fields": {"addr": ("Int128", IN_ADDR), "op": ("Op", IN_OP), "x": ("Byte", IN_X), "y": ("Byte", IN_Y),
                           "z": ("Byte", IN_Z), "imm": ("Byte", IN_IMM)}},
}
QUERY = {"root": ("Int128", IN_ROOT), "arg": ("Byte", IN_ARG)}
QUERY_FLAG = IN_QUERY
RUNS = 65536                                          # step() runs at most this many times
OUTPUT = {"values": "Byte", "classes": {"FAULT": FAULT, "TIMEOUT": TIMEOUT}}
