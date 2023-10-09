#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include "DFRobot_PH.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DS18B20.h>
#include "DHT.h"


#define DESIRED_TEMP 32
#define DESIRED_PH   4
#define PH_THRESHOLD 0.5
#define VALVE_TIME   2
#define ACT_TIME     20

float pH_value;
float temp;
float press;
float hum;
float sol_temp;

#define PIN_VIBR     2
#define PIN_HEAT     4
#define PIN_LEMON    5
#define PIN_WATER    6
#define PIN_BUTTON   8
#define PIN_LED      9
#define PIN_ACT1     10
#define PIN_ACT2     11
#define ONE_WIRE_BUS 12
#define PIN_DHT      13
#define PH_PIN       A1

Adafruit_BMP280 bmp; // I2C
OneWire oneWire(ONE_WIRE_BUS);
DS18B20 sensor(&oneWire);
DFRobot_PH ph;
#define DHTTYPE DHT11   // DHT 11
DHT dht(PIN_DHT, DHTTYPE);

bool print_flag = true;
int state;

bool get_button();
void set_output(int pin, bool status);
void end_loop();
float readPH();
float readSolTemp();
void printValues();

void setup() {

  Serial.begin(9600);

  pinMode(PIN_ACT1, OUTPUT);
  pinMode(PIN_ACT2, OUTPUT);
  pinMode(PIN_BUTTON, INPUT);
  pinMode(PIN_LED, OUTPUT);
  pinMode(PIN_HEAT, OUTPUT);
  pinMode(PIN_LEMON, OUTPUT);
  pinMode(PIN_WATER, OUTPUT);
  pinMode(PIN_VIBR, OUTPUT);

  unsigned status = bmp.begin();
  if (!status) Serial.println("BMP280 failure");
  else {
    /* Default settings from datasheet. */
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     /* Operating Mode. */
                    Adafruit_BMP280::SAMPLING_X8,     /* Temp. oversampling */
                    Adafruit_BMP280::SAMPLING_X16,    /* Pressure oversampling */
                    Adafruit_BMP280::FILTER_X16,      /* Filtering. */
                    Adafruit_BMP280::STANDBY_MS_500); /* Standby time. */
  }

  sensor.begin();
  sensor.setResolution(12);
    
  ph.begin();
  
  dht.begin();

  state = 0;
  if(print_flag) Serial.println("Started");

}



void loop() {

  switch (state) {
    
    // wait state
    case 0: {
      if(get_button()) state++;
      break;
    }
    
    // read temperature
    case 1: {

      sol_temp = readSolTemp();

      if(print_flag){
        Serial.print("Temperature read: ");
        Serial.println(sol_temp);
      }

      if(sol_temp>=DESIRED_TEMP) state=3;
      else{
        if(print_flag) Serial.println("Wrong temperature.");
        state = 2;
      }

      break;

    }
    
    // heat
    case 2: {

      sol_temp = readSolTemp();

      if(print_flag){
        Serial.print("Temperature read: ");
        Serial.println(sol_temp);
      }

      if(sol_temp>=DESIRED_TEMP) {
        state++;
      }
      break;

    }
    
    // read pH
    case 3: {

      pH_value = readPH();

      if(pH_value < DESIRED_PH - PH_THRESHOLD) state = 4;
      else if(pH_value > DESIRED_PH + PH_THRESHOLD) state = 5;
      else state = 6;

      break;

    }
    
    // increase pH
    case 4: {
      delay(VALVE_TIME*1000);
      state = 3;
      break;
    }
    
    // decrease pH
    case 5: {
      delay(VALVE_TIME*1000);
      state = 3;
      break;
    }
    
    // correct pH; wait for button
    case 6: {
      if(get_button()) state++;
      break;
    }
    
    // take measurements
    case 7: {

      pH_value = readPH();
      temp = bmp.readTemperature();
      press = bmp.readPressure();
      hum = dht.readHumidity();
      sol_temp = readSolTemp();

      printValues();

      state++;
      break;

    }
    
    // extend actuator
    case 8: {

      delay(ACT_TIME*1000);
      state++;
      break;

    }
    
    // retract actuator
    case 9: {

      delay(ACT_TIME*1000);
      state=0;
      break;

    }
    
    default:
      state = 0;
      break;
  }

  end_loop();

}


void end_loop() {

  if(print_flag){
    Serial.print("Current State:");
    Serial.println(state);
  }

  set_output(PIN_LED, state==0 || state == 6);
  set_output(PIN_HEAT, state==2);
  set_output(PIN_WATER, state==4);
  set_output(PIN_LEMON, state==5);
  set_output(PIN_VIBR, state==4 || state == 5);
  set_output(PIN_ACT1, state==9);
  set_output(PIN_ACT2, state==8);

  delay(500);

}

bool get_button() {
  return (bool)digitalRead(PIN_BUTTON);
}

void set_output(int pin, bool status) {
  if(pin==PIN_WATER || pin==PIN_LEMON || pin==PIN_HEAT) status=!status;
  digitalWrite(pin, status);
}

float readSolTemp() {
  sensor.requestTemperatures();
  while (!sensor.isConversionComplete());  // wait until sensor is ready
  return sensor.getTempC();
}

float readPH() {

  float voltagePH,temperature;

  delay(1000);

  temperature = readSolTemp();                   // read your temperature sensor to execute temperature compensation
  voltagePH = analogRead(PH_PIN)/1024.0*5000;          // read the ph voltage
  return ph.readPH(voltagePH,temperature);       // convert voltage to pH with temperature compensation

}

void printValues() {

  Serial.print("sample:");

  Serial.print(pH_value);
  Serial.print(",");

  Serial.print(temp);
  Serial.print(",");

  Serial.print(press);
  Serial.print(",");

  Serial.print(hum);
  Serial.print(",");

  Serial.println(sol_temp);

}