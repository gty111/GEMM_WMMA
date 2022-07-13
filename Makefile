CUTLASS_DIR ?= $(shell spack location -i cutlass)
INCLUDE_DIR = $(CUTLASS_DIR)/include $(shell pwd)
NAME ?= gemm
ARCH ?= sm_80
bin: $(NAME).cu
	@nvcc -arch=$(ARCH) $(addprefix -I,$(INCLUDE_DIR)) $(NAME).cu -o $(NAME)
run: bin
	@./$(NAME)
.PHONY: bin run