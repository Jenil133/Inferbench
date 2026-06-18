# TensorRT-LLM and Nsight Systems are supplied by NVIDIA's pinned runtime image.
ARG TRTLLM_IMAGE=nvcr.io/nvidia/tensorrt-llm/release:0.18.0
FROM ${TRTLLM_IMAGE}

WORKDIR /workspace/inferbench
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY examples ./examples
COPY docs ./docs
COPY scripts ./scripts
RUN python3 -m pip install --no-cache-dir --no-deps .

ENTRYPOINT ["inferbench"]
CMD ["--help"]
