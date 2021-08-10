
all:
	python test_cooley.py

.PHONY: clean
clean:
	rm -f *.c *.so *~

