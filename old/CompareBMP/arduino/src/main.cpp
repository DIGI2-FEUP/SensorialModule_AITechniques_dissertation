#include <Adafruit_BMP280.h>
#include <SPI.h>
#include <Wire.h>

Adafruit_BMP280 bmp; // use I2C interface
Adafruit_Sensor *bmp_temp = bmp.getTemperatureSensor();
Adafruit_Sensor *bmp_pressure = bmp.getPressureSensor();

void printTP(float t, float p);

void setup() {

  Serial.begin(57600);

  Serial.println(F("BMP280 Sensor event test"));

  unsigned status;
  status = bmp.begin();
  if (!status) {
    Serial.println(F("Could not find a valid BMP280 sensor, check wiring or "
                      "try a different address!"));
    Serial.print("SensorID was: 0x"); Serial.println(bmp.sensorID(),16);
    Serial.print("        ID of 0xFF probably means a bad address, a BMP 180 or BMP 085\n");
    Serial.print("   ID of 0x56-0x58 represents a BMP 280,\n");
    Serial.print("        ID of 0x60 represents a BME 280.\n");
    Serial.print("        ID of 0x61 represents a BME 680.\n");
    while (1) delay(10);
  }

  /* Default settings from datasheet. */
  bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,     /* Operating Mode. */
                  Adafruit_BMP280::SAMPLING_X2,     /* Temp. oversampling */
                  Adafruit_BMP280::SAMPLING_X16,    /* Pressure oversampling */
                  Adafruit_BMP280::FILTER_X16,      /* Filtering. */
                  Adafruit_BMP280::STANDBY_MS_500); /* Standby time. */

  bmp_temp->printSensorDetails();

}

void loop() {

  sensors_event_t temp_event, pressure_event;
  delay(1000);

  bmp_temp->getEvent(&temp_event);
  bmp_pressure->getEvent(&pressure_event);
  printTP(temp_event.temperature, pressure_event.pressure);

}

void printTP(float t, float p) {
  
  Serial.print(F("Temperature = "));
  Serial.print(t);
  Serial.println(" *C");

  Serial.print(F("Pressure = "));
  Serial.print(p);
  Serial.println(" hPa");

  Serial.println();

}