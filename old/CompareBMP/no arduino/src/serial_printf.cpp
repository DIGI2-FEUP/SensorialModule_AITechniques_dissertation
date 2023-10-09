#include "serial_printf.hpp"

#define BAUD_RATE 57600
#define BAUD_PRESCALE ((F_CPU / (BAUD_RATE * 16UL)) - 1)

void serial_begin(unsigned long baud) {
  UBRR0H = (unsigned char)(BAUD_PRESCALE >> 8);
  UBRR0L = (unsigned char)(BAUD_PRESCALE);
  UCSR0B |= (1 << TXEN0);
  UCSR0C |= (1 << UCSZ01) | (1 << UCSZ00);
}

int serial_putchar(char c, FILE *stream) {
  if (c == '\n') {
    serial_putchar('\r', stream);
  }
  loop_until_bit_is_set(UCSR0A, UDRE0);
  UDR0 = c;
  return 0;
}

void serial_print_float(float value, int decimalPlaces) {
  char buffer[32];
  dtostrf(value, 0, decimalPlaces, buffer);
  fputs(buffer, stdout);
}

void serial_print_int32(int32_t value) {
  char buffer[12];
  snprintf(buffer, sizeof(buffer), "%" PRId32, value);
  fputs(buffer, stdout);
}