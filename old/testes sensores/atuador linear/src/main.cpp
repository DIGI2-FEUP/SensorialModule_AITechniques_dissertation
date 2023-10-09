#include <Arduino.h>

// DRV8871 Motor Driver pins
const int LIN_POS = 9;
const int LIN_NEG = 8;

void setup() {

  pinMode(LIN_POS, OUTPUT);
  pinMode(LIN_NEG, OUTPUT);
  Serial.begin(9600);
  
  Serial.println("Waiting");
  // Extend the linear actuator
  digitalWrite(LIN_NEG, LOW); // Set direction (extend)
  digitalWrite(LIN_POS, LOW); // Set direction (extend)
  delay(5000); // Run for 2 seconds

}

void loop() {
  
  Serial.println("Extend");
  // Extend the linear actuator
  digitalWrite(LIN_NEG, LOW); // Set direction (extend)
  digitalWrite(LIN_POS, HIGH); // Set direction (extend)
  delay(20000); // Run for 2 seconds
  
  Serial.println("Pause");
  // Extend the linear actuator
  digitalWrite(LIN_NEG, LOW); // Set direction (extend)
  digitalWrite(LIN_POS, LOW); // Set direction (extend)
  delay(2000); // Run for 2 seconds
  
  Serial.println("Retract");
  // Extend the linear actuator
  digitalWrite(LIN_NEG, HIGH); // Set direction (extend)
  digitalWrite(LIN_POS, LOW); // Set direction (extend)
  delay(20000); // Run for 2 seconds
  
  Serial.println("Pause");
  // Extend the linear actuator
  digitalWrite(LIN_NEG, LOW); // Set direction (extend)
  digitalWrite(LIN_POS, LOW); // Set direction (extend)
  delay(2000); // Run for 2 seconds

}