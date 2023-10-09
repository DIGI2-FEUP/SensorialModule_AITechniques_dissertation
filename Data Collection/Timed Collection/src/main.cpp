#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <Adafruit_BMP280.h>
#include "DFRobot_PH.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DS18B20.h>
#include "DHT.h"

uint8_t sample_id = 0; // write 1 to reset the EEPROM memory, 0 to read from the EEPROM and another value to define de ID of the next sample
const uint8_t lastIdAddress = 100;

//#define DESIRED_TEMP 32
//#define DESIRED_PH   4
//#define PH_THRESHOLD 0.5
#define VALVE_TIME   2
#define ACT_TIME     8
//#define HEAT_TIME    2
#define HEAT_THRSH   1.5 //alterar
#define N_EQ_SAMPLES 7

float pH_value;
float temp;
float press;
float hum;
float sol_temp;

float prev_sol_temp=1000.0;
uint8_t temp_count;

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
bool initialization = true;
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
  
  set_output(PIN_ACT1, false);
  set_output(PIN_ACT2, false);
  set_output(PIN_LED, false);
  set_output(PIN_HEAT, false);
  set_output(PIN_LEMON, false);
  set_output(PIN_WATER, false);
  set_output(PIN_VIBR, false);

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
  temp_count = 0;

  if(initialization) {
    set_output(PIN_ACT1, true);
    set_output(PIN_ACT2, false);
    delay(ACT_TIME*1000+5000);
    set_output(PIN_ACT1, false);
  }

  if(sample_id == 1) {
    EEPROM.write(lastIdAddress, 0);
    sample_id = 0;
  }
  else if(sample_id==0) {
    sample_id = EEPROM.read(lastIdAddress)+1;
  }
  else {
    EEPROM.write(lastIdAddress, sample_id-1);
  }

  Serial.print("Next ID: ");
  Serial.println(sample_id);

  if(print_flag) Serial.println("Started");

}



void loop() {

  switch (state) {
    
    // wait state
    case 0: {
      if(get_button()) state++;
      break;
    }
    
    // decide if take another measurement, increase temperature or pour water
    case 1: {

/*       // if the temperature has been increased and that increase is not significant, it pours water
      if(temp_count == 1 && abs(sol_temp-prev_sol_temp) < HEAT_THRSH) {
        temp_count = 0;
        state = 2;
      }    
      // still not the N samples (take another sample)
      else  */if(temp_count < N_EQ_SAMPLES) {
        state = 4;
      }
      // increase temperature
      else {
        state = 3;
        temp_count = 0;
      }

      break;

    }
    
    // pour water
    case 2: {
      delay(VALVE_TIME*1000);
      state = 0;
      break;
    }
    
    // increase temperature
    case 3: {
      sol_temp = readSolTemp();
      if(abs(prev_sol_temp-sol_temp)>=HEAT_THRSH) state = 0;
      if(get_button()) {
        Serial.print("Previous temp:");
        Serial.print(prev_sol_temp);
        Serial.print("\tsol_temp: ");
        Serial.print(sol_temp);
        Serial.print("\tThreshold: ");
        Serial.print(HEAT_THRSH);
        Serial.print("\tDesired temp: ");
        Serial.println(HEAT_THRSH+prev_sol_temp);
      }
      break;
    }
    
    // start taking sample
    // take measurements
    case 4: {

      prev_sol_temp = sol_temp;

      pH_value = readPH();
      temp = bmp.readTemperature();
      press = bmp.readPressure();
      hum = dht.readHumidity();
      sol_temp = readSolTemp();

      printValues();

      EEPROM.write(lastIdAddress, sample_id);
      sample_id++;
      temp_count++;
      state++;

      break;

    }
    
    // extend actuator
    case 5: {
      delay(ACT_TIME*1000);
      state++;
      break;
    }
    
    // retract actuator
    case 6: {
      delay(ACT_TIME*1000+1000);
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

  static uint8_t prev_state=100;
  if(print_flag && state!=prev_state){
    Serial.print("Current State:");
    Serial.println(state);
    prev_state = state;
  }

  set_output(PIN_LED, state==0);
  set_output(PIN_HEAT, state==3);
  set_output(PIN_WATER, state==2);
  set_output(PIN_LEMON, false);
  set_output(PIN_VIBR, state==2 || state==3);
  set_output(PIN_ACT1, state==6);
  set_output(PIN_ACT2, state==5);

}

bool get_button() {
  return (bool)digitalRead(PIN_BUTTON);
}

void set_output(int pin, bool status) {
  if(pin==PIN_WATER || pin==PIN_LEMON || pin==PIN_HEAT || pin==PIN_ACT2) status=!status;
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

  Serial.print(sample_id);
  Serial.print(",");

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