#ifndef SERIAL_PRINT_HPP
#define SERIAL_PRINT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <avr/io.h>

void serial_begin(unsigned long baud);
int serial_putchar(char c, FILE *stream);
void serial_print_float(float value, int decimalPlaces);
void serial_print_int32(int32_t value);

#endif