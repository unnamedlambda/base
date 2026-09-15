"""The entry points of the artifacts this directory runs.

A generator fixes each entry's function index, and the artifact carries only
the functions, so these names live here instead of travelling with it. Keep
them in step with the generator that emits each artifact.
"""

ENTRIES = {
    # MlpCifarAlgorithm
    "mlp_cifar": {
        "fetchAdj": 17,
        "fetchDh": 16,
        "fetchDlog": 26,
        "fetchDw1": 18,
        "fetchDw2": 19,
        "fetchH": 14,
        "fetchLogits": 4,
        "fetchW1": 20,
        "fetchW2": 21,
        "fetchZ1": 15,
        "main": 1,
        "runAct": 6,
        "runAdj": 10,
        "runBwd": 23,
        "runBwdBlas": 28,
        "runDh": 9,
        "runDw1": 11,
        "runDw2": 8,
        "runFwd": 22,
        "runFwd1": 5,
        "runFwd2": 7,
        "runFwdBlas": 27,
        "runSgd1": 12,
        "runSgd2": 13,
        "runSoftmax": 24,
        "uploadBias": 25,
        "uploadOneHot": 3,
        "uploadX": 2,
    },
    # MlpCifarAlgorithm
    "ten_moe_dispatch": {
        "bindExperts": 4,
        "fetchGate": 3,
        "fetchOut": 7,
        "main": 1,
        "runExperts": 6,
        "runRouter": 2,
        "uploadGates": 5,
    },
    # MlpCifarAlgorithm
    "ten_qwen_block": {
        "capture": 13,
        "fetchDW2": 6,
        "fetchOut": 3,
        "main": 1,
        "replay": 14,
        "replay2": 15,
        "replay4": 16,
        "runBlock": 2,
        "runFwd": 4,
        "runFwdFused": 21,
        "runSynced": 12,
        "runTo30": 8,
        "runTo31": 17,
        "runTo34": 18,
        "runTo35": 19,
        "runTo37": 20,
        "runTo45": 9,
        "runTo60": 10,
        "runTo75": 11,
        "uploadDOut": 5,
        "uploadW2": 7,
    },
}


def entries(name):
    """The entry points of one artifact, by the file name it was emitted as."""
    try:
        return ENTRIES[name]
    except KeyError:
        raise KeyError(
            f"no entry points recorded for {name!r}; this module lists: "
            + ", ".join(sorted(ENTRIES))
        ) from None
