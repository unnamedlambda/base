"""The entry points of the artifacts this directory runs.

A generator fixes each entry's function index, and the artifact carries only
the functions, so these names live here instead of travelling with it. Keep
them in step with the generator that emits each artifact.
"""

ENTRIES = {
    # VitShip
    "vit_block": {
        "bwd": 11,
        "capture": 14,
        "captureBlas": 18,
        "captureChain": 4,
        "captureRow": 20,
        "captureStep": 16,
        "captureStepChain": 7,
        "fetch": 3,
        "fetchAny": 10,
        "main": 1,
        "reload": 13,
        "replay": 15,
        "replayBlas": 19,
        "replayChain": 5,
        "replayRow": 21,
        "replayStep": 17,
        "replayStepChain": 8,
        "run": 2,
        "seed": 9,
        "sgd": 12,
        "step": 6,
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
