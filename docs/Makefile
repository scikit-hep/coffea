html: source/modules
	cd .. && sed -i.bak s/docs\\\/source\\\/// README.rst && cd docs
	sphinx-build -M html source build
	cd .. && mv README.rst.bak README.rst && cd docs

source/modules: source/reference.rst
	sphinx-autogen source/reference.rst -t source/_templates -o source/modules

clean:
	rm -rf build source/modules source/api

.PHONY: clean
