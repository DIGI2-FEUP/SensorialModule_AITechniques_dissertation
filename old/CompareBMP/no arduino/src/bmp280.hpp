#include <avr/io.h>
#include <util/twi.h>
#include "serial_printf.hpp"
#include <stdlib.h>
#include <inttypes.h>
#include <Arduino.h>

extern "C" {
  #include "twi_master.h"
}

#define addBMP280 0x77

uint8_t checkIdBMP280(void);
int forceReading(void);
int checkIfReady(void);
void readMeas(double* ret);