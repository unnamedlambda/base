"""A schema's top-level messages, in order, as the harness's TOPS X-macro.

    python3 tops.py <benchmark.proto> > tops.h

Base's `decode` and `encode` name a message by its place in this list
(`Proto.schema`'s top-level messages, the file's order).
"""

import re
import sys

tops = re.findall(r"^message (\w+)", open(sys.argv[1]).read(), re.M)
print("#define TOPS " + " ".join(f"TOP({t})" for t in tops))
