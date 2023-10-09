#include <avr/io.h>
#include <util/twi.h>
#include "serial_printf.hpp"
#include <stdlib.h>
#include "bmp280.hpp"
#include <inttypes.h>

extern "C" {
  #include "twi_master.h"
}



int main(void) {

  serial_begin(57600);
  stdout = fdevopen(serial_putchar, NULL);

  tw_init(TW_FREQ_100K, true);

/*   uint8_t bmpID = checkIdBMP280();
  printf("hex id (58): %x\n\n", bmpID); */

  double meas[2];
  readMeas(meas);
  printf("temp: ");
  serial_print_float(meas[0],2);
  printf(" ºC\n");
  printf("press: ");
  serial_print_float(meas[1]/100,3);
  printf(" hPa\n");

  //while(1) {}

  return 0;

}