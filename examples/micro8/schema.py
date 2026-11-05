"""Input/output format of MICRO-8, for the decision-pipe compiler (no task rules)."""
from task import (IN_A, IN_ADDR, IN_ARG, IN_B, IN_DADDR, IN_DATA, IN_DVAL, IN_INSTR, IN_OP, IN_QUERY,
                  MAX_INSTR, OPS, TIMEOUT)

ENUMS = {"Op": OPS}                                   # HALT, LDI, LD, LDX, ADD, ... , CALL, RET
RECORDS = {
    "program": {"key": "addr", "indicator": IN_INSTR,
                "fields": {"addr": ("Int16", IN_ADDR), "op": ("Op", IN_OP), "arg": ("Byte", IN_ARG)}},
    "data": {"key": "daddr", "indicator": IN_DATA,
             "fields": {"daddr": ("Int8", IN_DADDR), "value": ("Byte", IN_DVAL)}},
}
QUERY = {"a": ("Byte", IN_A), "b": ("Byte", IN_B)}
QUERY_FLAG = IN_QUERY
RUNS = MAX_INSTR                                      # step() runs 16 times
OUTPUT = {"values": "Byte", "classes": {"TIMEOUT": TIMEOUT}}
