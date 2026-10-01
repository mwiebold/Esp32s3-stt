idf_component_register(SRCS "main.c" "mic.c" "oled.c" "oled_render.c" INCLUDE_DIRS "." REQUIRES tinyasr esp_partition spi_flash esp_timer esp_psram esp_driver_i2s esp_driver_i2c)
