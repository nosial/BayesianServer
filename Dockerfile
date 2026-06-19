FROM maven:3.9-eclipse-temurin-21-alpine AS builder

WORKDIR /build

COPY pom.xml .
COPY src ./src
COPY stopwords-json ./stopwords-json

RUN mvn -B package -DskipTests

FROM eclipse-temurin:21-jre-alpine

ARG BUILD_DATE
ARG REVISION
ARG VERSION=1.0.0

LABEL org.opencontainers.image.title="BayesianServer" \
      org.opencontainers.image.description="High-throughput HTTP API server for incremental Bayesian Naive Bayes multi-label text classification" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/nosial/BayesianServer" \
      org.opencontainers.image.license="MIT"

RUN addgroup -S bayesian && adduser -S bayesian -G bayesian

WORKDIR /app

COPY --from=builder /build/target/bayesian-server.jar /app/bayesian-server.jar

RUN mkdir -p /data/model /data/logs && chown -R bayesian:bayesian /app /data

USER bayesian

EXPOSE 8080

VOLUME ["/data/model", "/data/logs"]

STOPSIGNAL SIGTERM

ENTRYPOINT ["java", \
    "-XX:MaxRAMPercentage=75.0", \
    "-XX:+ExitOnOutOfMemoryError", \
    "-Dbayesian.log.dir=/data/logs", \
    "-jar", "/app/bayesian-server.jar", \
    "--model", "/data/model"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/health || exit 1
