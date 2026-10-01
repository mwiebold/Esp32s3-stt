idf_component_register(SRCS "tinyasr.c" "kernels.c" "tinyasr_lm.c" "tasr_nemo.c" "tasr_seg.c"
                       INCLUDE_DIRS "include" "."
                       PRIV_INCLUDE_DIRS "." PRIV_REQUIRES esp_timer)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function)

if(CONFIG_TASR_NEMO_FLOAT_STATS)
  target_compile_definitions(${COMPONENT_LIB} PRIVATE TASR_NEMO_FLOAT_STATS)
endif()
