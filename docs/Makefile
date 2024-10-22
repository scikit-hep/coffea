html: source/modules
	sphinx-build -M html source build

source/modules: source/reference.rst
	sphinx-autogen source/reference.rst -t source/_templates -o source/modules

clean:
	rm -rf build source/modules source/api

.PHONY: clean
