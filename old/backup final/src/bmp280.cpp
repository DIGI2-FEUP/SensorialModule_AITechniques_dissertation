#include "bmp280.hpp"
#include <util/atomic.h>

uint32_t dig_T1, dig_P1 = 0;
int32_t dig_T2, dig_T3, dig_P2, dig_P3, dig_P4, dig_P5, dig_P6, dig_P7, dig_P8, dig_P9 = 0;
int32_t t_fine;

uint8_t checkIdBMP280() {

  uint8_t ret = 0xD0;

  tw_master_transmit(addBMP280, &ret, 1, true);
  tw_master_receive(addBMP280, &ret, 1);

  return ret;

}

int forceReading(void) {

  uint8_t data[] = {0xF4, 0b01101101};

  tw_master_transmit(addBMP280, data, 2, true);

  return 0;

}

int checkIfReady(void) {

  uint8_t add = 0xF3;
  uint8_t data = 0x00;

/*   while(true) {

    tw_master_transmit(addBMP280, &add, 1, true);
    tw_master_receive(addBMP280, &data, 1);

    if( (data & (1<<3)) != 0 )
      break;

  } */

  //printf("Break done!\n");

  while(true) {

    tw_master_transmit(addBMP280, &add, 1, true);
    tw_master_receive(addBMP280, &data, 1);

    if( (data & (1<<3)) == 0 )
      return 1;

  }

}

void readCalib(void) {

  uint8_t add = 0x88;
  uint8_t data[26];

  tw_master_transmit(addBMP280, &add, 1, true);
  tw_master_receive(addBMP280, data, 26);

  dig_T1 = data[0];
  dig_T1 |= data[1] << 8;

  dig_T2 = data[2];
  dig_T2 |= data[3] << 8;

  dig_T3 = data[4];
  dig_T3 |= data[5] << 8;

  dig_P1 = data[6];
  dig_P1 |= data[7] << 8;

  dig_P2 = data[8];
  dig_P2 |= data[9] << 8;

  dig_P3 = data[10];
  dig_P3 |= data[11] << 8;

  dig_P4 = data[12];
  dig_P4 |= data[13] << 8;

  dig_P5 = data[14];
  dig_P5 |= data[15] << 8;

  dig_P6 = data[16];
  dig_P6 |= data[17] << 8;

  dig_P7 = data[18];
  dig_P7 |= data[19] << 8;

  dig_P8 = data[20];
  dig_P8 |= data[21] << 8;

  dig_P9 = data[22];
  dig_P9 |= data[23] << 8;

  return;

}

double calcTemp(int32_t adc_T) {

  cli();

  int32_t var1, var2, T, aux1, aux2, aux3, auxvar2;

  var1 = ((((adc_T>>3) - (dig_T1<<1))) * (dig_T2)) >> 11;

  aux1 = ((adc_T >> 4) - (dig_T1));
  aux2 = ((adc_T >> 4) - (dig_T1));
  aux3 = aux1 * aux2;
  aux3 >>= 12;
  auxvar2 = aux3 * dig_T3;
  var2 = auxvar2 >> 14;

  t_fine = var1 + var2;
  T = (t_fine * 5 + 128) >> 8;

  sei();

  return T/100.0;

}

uint32_t calcPress(int32_t adc_P) {

/*   int64_t var1, var2, p, aux;
  var1 = ((int64_t)t_fine) - 128000;
  var2 = var1 * var1;
  var2 *= (int64_t)dig_P6;
  aux = (var1*(int64_t)dig_P5);
  aux <<= 17;
  var2 += aux;
  aux = (((int64_t)dig_P4)<<35);
  var2 += aux;
  aux = (var1 * (int64_t)dig_P2);
  aux <<= 12;
  var1 *= var1;
  var1 *= (int64_t)dig_P3;
  var1 >>= 8;
  var1 += aux;
  aux = (((int64_t)1)<<47);
  var1 += aux;
  var1 *= ((int64_t)dig_P1);
  var1 >>= 33;
  if (var1 == 0)
  {
    return 0; // avoid exception caused by division by zero
  }
  p = 1048576-(int64_t)adc_P;
  p <<= 31;
  p -= var2;
  p *= 3125;
  p /= var1;
  aux = p >> 13;
  aux *= aux;
  var1 = ((int64_t)dig_P9)*aux;
  var1 >>= 25;
  var2 = (((int64_t)dig_P8) * p);
  var2 >>= 19;
  p = p + var1 + var2;
  p >>= 8;
  aux = (((int64_t)dig_P7)<<4);
  p += aux;
  return (uint32_t)p; */

  cli();

  int64_t var1, var2, p;
  int32_t aux;

  var1 = ((int64_t)t_fine) - 128000;

  var2 = var1 * var1 * (int64_t)dig_P6;
  var2 = var2 + ((var1 * (int64_t)dig_P5) << 17);
  var2 = var2 + (((int64_t)dig_P4) << 35);
  var1 = ((var1 * var1 * (int64_t)dig_P3));
  var1 >>= 8;
  var1 += ((var1 * (int64_t)dig_P2) << 12);
  var1 =
      (((((int64_t)1) << 47) + var1)) * ((int64_t)dig_P1);

  aux = var1>>32;
  printf("var1: %lx", aux);
  aux = var1;
  printf("%lx\n", aux);
  var1 >>= 33;

  if (var1 == 0) {
    return 0; // avoid exception caused by division by zero
    printf("var1 = 0\n");
  }

  p = 1048576 - adc_P;
  p = (((p << 31) - var2) * 3125) / var1;
  var1 = (((int64_t)dig_P9) * (p >> 13) * (p >> 13)) >> 25;
  var2 = (((int64_t)dig_P8) * p) >> 19;

  p = ((p + var1 + var2) >> 8) + (((int64_t)dig_P7) << 4);

  sei();

  return (float)p;

}



void readMeas(double* ret) {

  forceReading();
  readCalib();
  checkIfReady();

  uint8_t add = 0xF7;
  uint8_t data[6];

  tw_master_transmit(addBMP280, &add, 1, true);
  tw_master_receive(addBMP280, data, 6);

  uint8_t temp_xlsb, temp_lsb, temp_msb, press_xlsb, press_lsb, press_msb;
  press_msb = data[0];
  press_lsb = data[1];
  press_xlsb = data[2];
  temp_msb = data[3];
  temp_lsb = data[4];
  temp_xlsb = data[5];

  int32_t temp = 0, press = 0;

  temp |= static_cast<uint32_t>(temp_xlsb) >> 4;
  temp |= static_cast<uint32_t>(temp_lsb) << 4;
  temp |= static_cast<uint32_t>(temp_msb) << 12;

  press |= static_cast<uint32_t>(press_xlsb) >> 4;
  press |= static_cast<uint32_t>(press_lsb) << 4;
  press |= static_cast<uint32_t>(press_msb) << 12;

  ret[0] = calcTemp(temp);
  ret[1] = calcPress(press)/25600.0;

  return;

}