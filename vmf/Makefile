F=-L/usr/local/lib -lgmp -lmpfr -lmpfrcpp -lntl -DDSFMT_MEXP=44497
OBJS = o/duhamel.o o/dSFMT.o o/conform.o o/newbessel.o o/vmf_distribution.o 
CC = g++
DEBUG = -g
CFLAGS = -c $(DEBUG) $F
LFLAGS = $(DEBUG) $F

conform: dircheck $(OBJS)
	$(CC) $(OBJS) -o conform $(LFLAGS)

o/duhamel.o: Duhamel.cpp
	$(CC) Duhamel.cpp -o o/duhamel.o $(CFLAGS)

o/newbessel.o : newbessel.cc
	$(CC) newbessel.cc -o o/newbessel.o $(CFLAGS)

o/vmf_distribution.o : vmf_distribution.cpp
	$(CC)  vmf_distribution.cpp -o o/vmf_distribution.o $(CFLAGS)

o/dSFMT.o: dSFMT/dSFMT.c
	$(CC) ./dSFMT/dSFMT.c -DDSFMT_MEXP=44497 -o o/dSFMT.o $(CFLAGS) 

o/conform.o: conformation.cpp  
	$(CC) conformation.cpp -o o/conform.o $(CFLAGS)
clean:
	\rm o/*.o *~ conform

tar:
	tar jcvf vmf.tar.bz2 newbessel.h newbessel.cc vmf_distribution.hpp vmf_distribution.cpp test.cpp \
		Makefile  README 
dircheck:
	if [ ! -d o ]; then mkdir o; fi;