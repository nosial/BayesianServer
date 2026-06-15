.PHONY: all build test clean run run-debug help

JAR_NAME   = bayesian-server
JAR_PATH   = target/$(JAR_NAME).jar
MAIN_CLASS = net.nosial.bayesian_server.Program

all: build

build:
	mvn -B package -DskipTests

test:
	mvn -B test

clean:
	mvn -B clean

run: $(JAR_PATH)
	java -jar $(JAR_PATH)

## run-debug  — Run with debug logging
run-debug: $(JAR_PATH)
	java -Dlogback.level=DEBUG -jar $(JAR_PATH)

$(JAR_PATH):
	mvn -B package -DskipTests