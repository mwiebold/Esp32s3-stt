idf_component_register(SRCS "tinyasr.c" "kernels.c" "tinyasr_lm.c" "tasr_nemo.c" "tasr_seg.c"
                       INCLUDE_DIRS "include" "."
                       PRIV_INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function)
