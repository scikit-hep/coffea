# check whether parallel jobs were enabled for make
_PARALLEL_JOBS_FLAG = $(filter -j%,$(MAKEFLAGS))

# number of processors on this machine (always non-empty)
_NPROC := $(shell nproc)

# number given by jobs flag (empty if `-j` absent or does not have a following N)
_NJOBS_REQ = $(patsubst -j%,%,$(_PARALLEL_JOBS_FLAG))

# or returns first instance of non-empty string
# if `-j[N]` given on command line, return N or number of processors if N absent
# otherwise use default of 1
NJOBS = $(if $(_PARALLEL_JOBS_FLAG),$(or $(_NJOBS_REQ),$(_NPROC)),1)

# build options
# 	source              : directory containing source files
# 	build/html          : output directory to write files into
# 												extra sub-directory to keep jupyter-cache within build but outside site
#   --builder html      : choose to build the html target
# 	--warning-file FILE : dump warnings into build/warnings.txt for later review
# 	--nitpicky          : warn on every missing link
# 	--jobs NJOBS        : number of parallel jobs to spawn
OPTIONS=source build/html --builder html --warning-file build/warnings.txt --nitpicky --jobs $(NJOBS)

html: source/modules
	sphinx-build $(OPTIONS)

# construct ReST files for the doc website from the module docstrings
# sphinx-autogen
# 	source/api_reference.md : root source file containing autosummary and/or autodoc
# 	-t source/_templates    : define directory with custom templates
# 	-o source/modules       : output directory where generated rst should go
source/modules: source/api_reference.md
	sphinx-autogen source/api_reference.md -t source/_templates -o source/modules

# remove generated files
clean:
	rm -rf build source/modules source/api

# auto-build the site after the source changes
# this is only helpful when doing a lot of writing within the source/ directory.
# since updates to the docstrings need to pass through sphinx-autogen
# before getting to sphinx-build, they do not get picked up in the auto-build
# --ignore : glob expression to ignore files, ignoring emacs and vim buffer files
watch:
	sphinx-autobuild --ignore "*.swp" --ignore "~*" $(OPTIONS)

.PHONY: clean
