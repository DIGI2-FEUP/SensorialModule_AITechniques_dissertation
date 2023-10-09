#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include "Adafruit_TCS34725.h"

/* Example code for the Adafruit TCS34725 breakout library */

/* Connect SCL    to analog 5
   Connect SDA    to analog 4
   Connect VDD    to 3.3V DC
   Connect GROUND to common ground */

/* Initialise with default values (int time = 2.4ms, gain = 1x) */
// Adafruit_TCS34725 tcs = Adafruit_TCS34725();

/* Initialise with specific int time and gain values */
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_614MS, TCS34725_GAIN_1X);

#define PIN_BUTTON   8
#define PIN_LED      9
#define CYCLES       5

bool get_button();
void set_output(int pin, bool status);

uint16_t r, g, b, c, colorTemp, lux;
uint16_t r2, g2, b2, c2, colorTemp2, lux2;

void setup(void) {
  Serial.begin(9600);

  if (tcs.begin()) {
    Serial.println("Found sensor");
  } else {
    Serial.println("No TCS34725 found ... check your connections");
    while (1);
  }

  set_output(PIN_LED, true);

}

void loop(void) {

  if(get_button()) {

    set_output(PIN_LED, false);

    r2=0;
    g2=0;
    b2=0;
    c2=0;
    colorTemp2=0;
    lux2=0;

    for(int i=0; i<CYCLES; i++) {

      tcs.getRawData(&r, &g, &b, &c);
      colorTemp = tcs.calculateColorTemperature_dn40(r, g, b, c);
      lux = tcs.calculateLux(r, g, b);

      r2+=r;
      g2+=g;
      b2+=b;
      c2+=c;
      colorTemp2+=colorTemp;
      lux2+=lux;
    
    }

    r=r2/CYCLES;
    g=g2/CYCLES;
    b=b2/CYCLES;
    c=c2/CYCLES;
    colorTemp=colorTemp2/CYCLES;
    lux=lux2/CYCLES;

    Serial.print("Color Temp: "); Serial.print(colorTemp, DEC); Serial.print(" K - ");
    Serial.print("Lux: "); Serial.print(lux, DEC); Serial.print(" - ");
    Serial.print("R: "); Serial.print(r, DEC); Serial.print(" ");
    Serial.print("G: "); Serial.print(g, DEC); Serial.print(" ");
    Serial.print("B: "); Serial.print(b, DEC); Serial.print(" ");
    Serial.print("C: "); Serial.print(c, DEC); Serial.print(" ");
    Serial.println(" ");

    set_output(PIN_LED, true);

  }

}

bool get_button() {
  return (bool)digitalRead(PIN_BUTTON);
}

void set_output(int pin, bool status) {
  digitalWrite(pin, status);
}