#include <avr/io.h>
#include <util/twi.h>
#include "serial_printf.hpp"
#include <stdlib.h>
#include "bmp280.hpp"
#include <inttypes.h>
#include <Arduino.h>

extern "C" {
  #include "twi_master.h"
}



int main(void) {

  init();

  serial_begin(57600);
  stdout = fdevopen(serial_putchar, NULL);

/*   Serial.begin(57600);
  Serial.println("here"); */

  tw_init(TW_FREQ_100K, true);

  double meas[2];

  readMeas(meas);

  printf("temp: %.2lf *C\n");
  printf("press: %.2lf hPa\n");

/*   Serial.print("temp: ");
  Serial.print(meas[0]);
  Serial.println(" *C");

  Serial.print("press: ");
  Serial.print(meas[0]);
  Serial.println(" hPa"); */

  //while(1) {}

  return 0;

}