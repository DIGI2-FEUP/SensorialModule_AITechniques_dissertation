/*!
 * @file DFRobot_PH_EC.h
 * @brief This is the sample code for The Mixed use of two sensors: 
 * @n 1. Gravity: Analog pH Sensor / Meter Kit V2, SKU:SEN0161-V2
 * @n 2. Analog Electrical Conductivity Sensor / Meter Kit V2 (K=1.0), SKU: DFR0300.
 * @n In order to guarantee precision, a temperature sensor such as DS18B20 is needed, to execute automatic temperature compensation.
 * @n Serial Commands:
 * @n   PH Calibration：
 * @n    enterph -> enter the calibration mode
 * @n    calph   -> calibrate with the standard buffer solution, two buffer solutions(4.0 and 7.0) will be automaticlly recognized
 * @n    exitph  -> save the calibrated parameters and exit from calibration mode
 * @n   EC Calibration：
 * @n    enterph -> enter the PH calibration mode
 * @n    calph   -> calibrate with the standard buffer solution, two buffer solutions(4.0 and 7.0) will be automaticlly recognized
 * @n    exitph  -> save the calibrated parameters and exit from PH calibration mode
 *
 * @copyright   Copyright (c) 2010 DFRobot Co.Ltd (http://www.dfrobot.com)
 * @license     The MIT License (MIT)
 * @author [Jiawei Zhang](jiawei.zhang@dfrobot.com)
 * @version  V1.0
 * @date  2018-11-06
 * @url https://github.com/DFRobot/DFRobot_PH
 */

#include "DFRobot_PH.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DS18B20.h>

#define PH_PIN A1
#define ONE_WIRE_BUS 12
float  voltagePH,voltageEC,phValue,ecValue,temperature = 25;
DFRobot_PH ph;
OneWire oneWire(ONE_WIRE_BUS);
DS18B20 sensor(&oneWire);

bool readSerial(char result[]);
bool calibration_flag = false;
float readTemperature();

void setup()
{
    Serial.begin(9600);  
    ph.begin();

    sensor.begin();
    sensor.setResolution(12);
}

void loop()
{
    char cmd[10];
    static unsigned long timepoint = millis();
    if(millis()-timepoint>1000U){                            //time interval: 1s
        timepoint = millis();
        temperature = readTemperature();                   // read your temperature sensor to execute temperature compensation
        voltagePH = analogRead(PH_PIN)/1024.0*5000;          // read the ph voltage
        phValue    = ph.readPH(voltagePH,temperature);       // convert voltage to pH with temperature compensation
        Serial.print("pH:");
        Serial.println(phValue,2);
    }
    if(readSerial(cmd)){
        strupr(cmd);
        if(strstr(cmd,"PH")){
            ph.calibration(voltagePH,temperature,cmd);       //PH calibration process by Serail CMD
        }
    }
}

int i = 0;
bool readSerial(char result[]){
    while(Serial.available() > 0){
        char inChar = Serial.read();
        if(inChar == '\n'){
             result[i] = '\0';
             Serial.flush();
             i=0;
             return true;
        }
        if(inChar != '\r'){
             result[i] = inChar;
             i++;
        }
        delay(1);
    }
    return false;
}

float readTemperature()
{
    sensor.requestTemperatures();
    while (!sensor.isConversionComplete());
    float ret = sensor.getTempC();
    Serial.print("Temperature:");
    Serial.println(ret,2);
    return ret;

}