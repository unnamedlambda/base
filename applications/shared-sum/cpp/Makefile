# The C++ host for `shared_sum`, against base's C ABI.
#
#   make run                      build libbase (debug), the example, and run it
#   make run PROFILE=release      the same against a release libbase
#   make run THREADS=32           more threads than cores
#
# The artifact is the one `lake query algorithmLib/artifacts` (or any cargo
# build of `lean-artifacts`) writes; ARTIFACT points elsewhere. THREADS
# defaults to the number of hardware threads.

ROOT     := $(abspath ../../..)
PROFILE  ?= debug
LIBDIR   := $(ROOT)/target/$(PROFILE)
ARTIFACT ?= $(ROOT)/lean/.lake/build/artifacts/shared_sum.cbor
THREADS  ?=
CXX      ?= g++
CXXFLAGS ?= -std=c++20 -O2 -Wall -Wextra -pthread

# The command that builds libbase; a caller that caps cargo's memory passes
# its own.
CARGO    ?= cargo
CARGO_PROFILE := $(if $(filter release,$(PROFILE)),--release,)

shared_sum: main.cpp $(ROOT)/base/include/base.h $(ROOT)/base/include/base.hpp $(LIBDIR)/libbase.so
	$(CXX) $(CXXFLAGS) -I$(ROOT)/base/include main.cpp -o $@ \
	  -L$(LIBDIR) -lbase -Wl,-rpath,$(LIBDIR)

$(LIBDIR)/libbase.so: FORCE
	cd $(ROOT) && $(CARGO) build -p base $(CARGO_PROFILE) $(CARGO_FLAGS)

run: shared_sum
	./shared_sum $(ARTIFACT) $(THREADS)

clean:
	rm -f shared_sum

FORCE:
.PHONY: run clean FORCE
