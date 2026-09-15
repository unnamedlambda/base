"""The entry points of the artifacts this directory runs.

A generator fixes each entry's function index, and the artifact carries only
the functions, so these names live here instead of travelling with it. Keep
them in step with the generator that emits each artifact.
"""

ENTRIES = {
    # GptOssAlgorithm
    "gptoss_attn": {
        "fetchAtt": 6,
        "fetchQkv": 5,
        "fetchX": 4,
        "main": 1,
        "step": 3,
        "uploadStep": 2,
    },
    # GptOssAlgorithm
    "gptoss_layer": {
        "main": 1,
    },
    # GptOssAlgorithm
    "gptoss_moe": {
        "bindExperts": 2,
        "fetchOut": 6,
        "main": 1,
        "runExperts": 5,
        "uploadGates": 4,
        "uploadX": 3,
    },
    # GptOssDecode
    "gptoss_decode": {
        "main": 1,
    },
    # TokenizerTest
    "tokenizer_test": {
        "main": 1,
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
