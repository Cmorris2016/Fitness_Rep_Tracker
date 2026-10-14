/*
  MPU-6050 repetition counter

  Strap the sensor to the moving body segment (forearm for curls, upper arm
  for presses, thigh for squats). At power-up, hold the START position still
  for ~2 seconds while it calibrates. After that, each time the segment tilts
  away from the start position by REP_UP_DEG and comes back within
  REP_DOWN_DEG, one rep is counted.

  Serial commands (115200 baud):
    c  recalibrate the start position
    r  reset the rep count

  Open Tools > Serial Plotter to watch the angle and rep count live.

  Wiring (I2C):
    MPU-6050   Arduino Mega 2560   Arduino Uno/Nano   ESP32
    VCC        5V                  5V or 3.3V          3.3V
    GND        GND                 GND                 GND
    SDA        20 (SDA)            A4                  GPIO 21
    SCL        21 (SCL)            A5                  GPIO 22
*/

#include <Wire.h>
#include <math.h>

const uint8_t MPU_ADDR = 0x68;  // 0x69 if the AD0 pin is pulled high

// Tune these per exercise.
const float REP_UP_DEG = 60.0;            // tilt needed to count as the "top" of a rep
const float REP_DOWN_DEG = 25.0;          // must return within this to finish the rep
const unsigned long MIN_REP_MS = 500;     // ignore reps faster than this (bounce/jitter)

const unsigned long SAMPLE_MS = 20;       // 50 Hz
const float LPF_ALPHA = 0.2;              // 0..1, lower = smoother but laggier
const int CALIBRATION_SAMPLES = 100;      // ~2 s at 50 Hz

float ref[3];        // gravity direction in the start position
float filt[3];       // low-pass filtered acceleration
bool atTop = false;
int reps = 0;
unsigned long lastRepMs = 0;
unsigned long lastSampleMs = 0;

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// Reads acceleration in g (default +/-2 g range: 16384 LSB per g).
bool readAccel(float a[3]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)6) != 6) return false;
  for (int i = 0; i < 3; i++) {
    int16_t raw = (Wire.read() << 8) | Wire.read();
    a[i] = raw / 16384.0;
  }
  return true;
}

float magnitude(const float v[3]) {
  return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

// Angle in degrees between the current gravity direction and the start position.
float tiltFromStart() {
  float dot = filt[0] * ref[0] + filt[1] * ref[1] + filt[2] * ref[2];
  float c = dot / (magnitude(filt) * magnitude(ref) + 1e-6);
  c = constrain(c, -1.0, 1.0);
  return acos(c) * 180.0 / PI;
}

void calibrate() {
  Serial.println("# Calibrating: hold the start position still...");
  float sum[3] = {0, 0, 0};
  float a[3];
  int n = 0;
  while (n < CALIBRATION_SAMPLES) {
    if (readAccel(a)) {
      for (int i = 0; i < 3; i++) sum[i] += a[i];
      n++;
    }
    delay(SAMPLE_MS);
  }
  for (int i = 0; i < 3; i++) {
    ref[i] = sum[i] / n;
    filt[i] = ref[i];
  }
  atTop = false;
  Serial.println("# Calibrated. Start your set.");
  // Machine-readable settings, used by the web dashboard to draw thresholds.
  Serial.print("# config up=");
  Serial.print(REP_UP_DEG, 1);
  Serial.print(" down=");
  Serial.println(REP_DOWN_DEG, 1);
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000);

  writeRegister(0x6B, 0x00);  // PWR_MGMT_1: wake up
  writeRegister(0x1A, 0x03);  // CONFIG: 44 Hz digital low-pass filter
  writeRegister(0x1C, 0x00);  // ACCEL_CONFIG: +/-2 g

  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("# MPU-6050 not found. Check wiring and I2C address.");
    while (true) delay(1000);
  }

  calibrate();
}

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 'c') calibrate();
    if (cmd == 'r') { reps = 0; atTop = false; Serial.println("# Count reset."); }
  }

  unsigned long now = millis();
  if (now - lastSampleMs < SAMPLE_MS) return;
  lastSampleMs = now;

  float a[3];
  if (!readAccel(a)) return;
  for (int i = 0; i < 3; i++) filt[i] += LPF_ALPHA * (a[i] - filt[i]);

  float angle = tiltFromStart();

  // Hysteresis: go past the top threshold, then come back below the bottom one.
  if (!atTop && angle > REP_UP_DEG) {
    atTop = true;
  } else if (atTop && angle < REP_DOWN_DEG) {
    atTop = false;
    if (now - lastRepMs >= MIN_REP_MS) {
      reps++;
      lastRepMs = now;
    }
  }

  // "label:value" pairs are understood by the Arduino Serial Plotter.
  Serial.print("angle:");
  Serial.print(angle, 1);
  Serial.print(",reps:");
  Serial.println(reps);
}
