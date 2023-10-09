#include "Adafruit_APDS9960.h"
#include <Adafruit_BMP280.h>
#include <SPI.h>
#include <Wire.h>
#include "serial_printf.hpp"

Adafruit_APDS9960 apds;

Adafruit_BMP280 bmp; // use I2C interface
Adafruit_Sensor *bmp_temp = bmp.getTemperatureSensor();
Adafruit_Sensor *bmp_pressure = bmp.getPressureSensor();

unsigned state;

//the pin that the interrupt is attached to
#define INT_PIN 2
#define relay 3

void printColorData(uint16_t r, uint16_t g, uint16_t b, uint16_t c);
void printTP(float t, float p);

void setup() {

  init();

  serial_begin(57600);
  stdout = fdevopen(serial_putchar, NULL);

  Serial.begin(57600);
  pinMode(INT_PIN, INPUT_PULLUP);
  pinMode(relay, OUTPUT);

  if(!apds.begin()){
    printf("failed to initialize device! Please check your wiring.\n");
  }
  else printf("Device initialized!\n");

  //enable color sensign mode
  apds.enableColor(true);

  printf("BMP280 Sensor event test\n");

  unsigned status;
  status = bmp.begin();
  if (!status) {
    printf("Could not find a valid BMP280 sensor, check wiring or try a different address!\n");
    printf("SensorID was: 0x"); printf("%x\n", bmp.sensorID());
    printf("        ID of 0xFF probably means a bad address, a BMP 180 or BMP 085\n");
    printf("   ID of 0x56-0x58 represents a BMP 280,\n");
    printf("        ID of 0x60 represents a BME 280.\n");
    printf("        ID of 0x61 represents a BME 680.\n");
    while (1) delay(10);
  }

  /* Default settings from datasheet. */
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     /* Operating Mode. */
                  Adafruit_BMP280::SAMPLING_X2,     /* Temp. oversampling */
                  Adafruit_BMP280::SAMPLING_X16,    /* Pressure oversampling */
                  Adafruit_BMP280::FILTER_X16,      /* Filtering. */
                  Adafruit_BMP280::STANDBY_MS_500); /* Standby time. */

  bmp_temp->printSensorDetails();

  state = 1;

}

void loop() {

  printf("state: %d\n", state);

  switch (state) {

    case 1:

      //create some variables to store the color data in
      uint16_t r, g, b, c;
      sensors_event_t temp_event, pressure_event;
      delay(5000);

      state = 2;
      break;

    case 2:

      bmp_temp->getEvent(&temp_event);
      bmp_pressure->getEvent(&pressure_event);
      printTP(temp_event.temperature, pressure_event.pressure);
      state = 3;
      
    case 3:

      delay(1000);
      state = 4;
      
    case 4:
      
      //activation and deactivation of relay
      digitalWrite(relay, HIGH);
      delay(1000); 
      digitalWrite(relay, LOW); 
      delay(1000);

      state = 5;
      
    case 5:
      
      //wait for color data to be ready
      while(!apds.colorDataReady()){
        delay(5);
      }

      //get the data and print the different channels
      apds.getColorData(&r, &g, &b, &c);
      printColorData(r, g, b, c);

      //print the proximity reading when the interrupt pin goes low
      if(!digitalRead(INT_PIN)){
        printf("%u\n", apds.readProximity());

        //clear the interrupt
        apds.clearInterrupt();
      }
 
      state = 1;
    
    default:
      state = 1;
      break;
  }

}

void printTP(float t, float p) {
  
  printf("Temperature = %.2f *C\n", t);
  printf("Pressure = %.2f hPa\n", p);

}

void printColorData(uint16_t r, uint16_t g, uint16_t b, uint16_t c) {

  printf("red: %u", r);
  printf(" green: %u", g);
  printf(" blue: %u", b);
  printf(" clear: %u\n", c);

}