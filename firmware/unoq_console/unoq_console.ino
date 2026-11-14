/*
  ATC operator console for the Arduino UNO Q STM32U585 MCU (T-131, spec 5.2).

  Hardware (matches the console wiring diagram, all modules on the 3V3 / GND rails):
    D2   Clutch button  KY-004 (hold-to-control dead-man's clutch)
    D3   Alert buzzer   KY-006 passive (driven with tone())
    D4/5 Bay 1 LED      KY-011 R/G      D6/7 Bay 2 LED  KY-011 G/R
    D8/9 Bay 3 LED      KY-011 R/G      D10/11 Bay 4 LED KY-011 G/R
    D12  E-stop button  KY-004 (normally open)
    D13  Judge fault rig switch to GND (optional; unwired = never latches)
    A0/A1 Bay-select joystick KY-023 VRx/VRy, A2 its SW (optional)
    Onboard 8x13 LED matrix: one 3-column x 8-row tile per bay + a status column.

  - Polls clutch, E-stop, joystick switch and the fault rig at 1 kHz (5 ms debounce); samples
    the joystick at 100 Hz. Flicking it up/right/down/left selects bay 1/2/3/4.
  - Sends text lines to the Linux MPU through arduino-router: Bridge.notify("console_line", ..)
      BTN:BAY:<1-4>  CLUTCH:<0|1>  ESTOP:<0|1>  FAULT_RIG:LATCHED
      HB:<bay 0-4>:<clutch>:<estop>:<rig>   (20 Hz heartbeat)
      JOY:<x>:<y>                           (only with JOY_DEBUG; ignored by the host parser)
  - Serves Bridge "console_cmd" for MPU -> MCU lines:
      CMD:BAY_STATE:<1-4>:<NOMINAL|STALLED|QUEUED|RESCUE|VALIDATING|AUTO|OOD>
      CMD:RESET_RIG
  - Mirrors each bay state on its bi-colour LED and the matrix; beeps when a bay needs a human
    (STALLED / OOD) and alarms while the E-stop is active (all bay LEDs forced red).

  Fault rig is strictly one-way: the first rising edge emits FAULT_RIG:LATCHED once, then every
  further toggle is ignored until CMD:RESET_RIG. Python side: src/teleop/unoq/mcu_bridge.py.

  Build: arduino-cli compile --fqbn arduino:zephyr:unoq firmware/unoq_console
*/

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <zephyr/kernel.h>

// ---------------------------------------------------------------- wiring (edit to match rig)
static const uint8_t PIN_CLUTCH = 2;
static const uint8_t PIN_BUZZER = 3;
static const uint8_t PIN_ESTOP = 12;
static const uint8_t PIN_FAULT_RIG = 13;  // switch to GND, closed = flap raised
static const uint8_t PIN_JOY_X = A0;
static const uint8_t PIN_JOY_Y = A1;
static const uint8_t PIN_JOY_SW = A2;     // switch to GND, closed = pressed
static const uint8_t PIN_LED_RED[4] = {4, 7, 8, 11};
static const uint8_t PIN_LED_GREEN[4] = {5, 6, 9, 10};

// KY-004 modules differ: most pull S high and short it to GND when pressed (active low). If
// CLUTCH:1 / ESTOP:1 show up while the buttons are released, flip this to false.
#define BUTTON_ACTIVE_LOW true
// The KY-004 E-stop is a normally-open push button: pressed = stop. A real normally-closed
// mushroom switch (fail-safe on a cut wire) would set this to true.
#define ESTOP_NORMALLY_CLOSED false

// Joystick orientation: which way is "up" (bay 1) depends on how the KY-023 is mounted. Set
// JOY_DEBUG to 1 to stream JOY:<x>:<y> and read the raw values while you push the stick.
#define JOY_SWAP_XY false
#define JOY_INVERT_X false  // false: higher VRx = right (bay 2)
#define JOY_INVERT_Y true   // true: lower VRy = up (bay 1)
#define JOY_DEBUG 0

#define BUZZER_ENABLED true

// ---------------------------------------------------------------- timing
static const uint32_t POLL_US = 1000;          // 1 kHz input scan
static const uint8_t DEBOUNCE_MS = 5;          // level must be stable this long
static const uint32_t JOY_MS = 10;             // 100 Hz joystick sampling
static const uint32_t HEARTBEAT_MS = 50;       // 20 Hz
static const uint32_t FRAME_MS = 20;           // 50 Hz LED / buzzer refresh
static const uint32_t LINK_TIMEOUT_MS = 2000;  // no MPU command for this long -> link lost

// Joystick thresholds on the 10-bit ADC, measured from the centre captured at boot.
static const int JOY_FIRE = 300;       // deflection that selects a bay (~60% of half travel)
static const int JOY_REARM = 120;      // stick must come back inside this before the next flick
static const uint8_t JOY_CONFIRM = 3;  // consecutive samples past JOY_FIRE (30 ms)

// ---------------------------------------------------------------- debounced inputs
struct Debounced {
  uint8_t pin;
  bool activeLow;
  bool stable;  // debounced "closed / pressed" level
  bool raw;
  uint8_t count;
};

static Debounced clutchIn, estopIn, rigIn, joySwIn;

static bool readClosed(const Debounced &d) {
  return (digitalRead(d.pin) == LOW) == d.activeLow;
}

static void initInput(Debounced &d, uint8_t pin, bool activeLow) {
  pinMode(pin, INPUT_PULLUP);
  d.pin = pin;
  d.activeLow = activeLow;
  d.raw = d.stable = readClosed(d);
  d.count = 0;
}

// Returns true when the debounced level changed this tick.
static bool updateInput(Debounced &d) {
  bool now = readClosed(d);
  if (now != d.raw) {
    d.raw = now;
    d.count = 0;
    return false;
  }
  if (d.stable != now && ++d.count >= DEBOUNCE_MS) {
    d.stable = now;
    d.count = 0;
    return true;
  }
  return false;
}

static bool estopActive() {
  return ESTOP_NORMALLY_CLOSED ? !estopIn.stable : estopIn.stable;
}

static bool clutchHeld() { return clutchIn.stable && !estopActive(); }

// ---------------------------------------------------------------- console state
static uint8_t selectedBay = 0;  // 0 = none yet
static bool rigLatched = false;

static void sendLine(const char *line) { Bridge.notify("console_line", MsgPack::str_t(line)); }

static void sendBit(const char *prefix, bool v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%s:%d", prefix, v ? 1 : 0);
  sendLine(buf);
}

static void sendHeartbeat() {
  char buf[24];
  snprintf(buf, sizeof(buf), "HB:%u:%d:%d:%d", selectedBay, clutchIn.stable ? 1 : 0,
           estopActive() ? 1 : 0, rigLatched ? 1 : 0);
  sendLine(buf);
}

static void sendBaySelect() {
  char buf[12];
  snprintf(buf, sizeof(buf), "BTN:BAY:%u", selectedBay);
  sendLine(buf);
}

// ---------------------------------------------------------------- buzzer
static uint32_t chirpUntilMs = 0;
static unsigned int chirpHz = 0;
static unsigned int buzzerHz = 0;  // what tone() is currently playing

static void chirp(unsigned int hz, uint32_t ms) {
  chirpHz = hz;
  chirpUntilMs = millis() + ms;
}

static void setBuzzer(unsigned int hz) {
  if (!BUZZER_ENABLED) hz = 0;
  if (hz == buzzerHz) return;  // tone() restarts its timer, so only touch it on a change
  buzzerHz = hz;
  if (hz) {
    tone(PIN_BUZZER, hz);
  } else {
    noTone(PIN_BUZZER);
  }
}

// ---------------------------------------------------------------- joystick
static int joyCx = 512, joyCy = 512;
static bool joyArmed = true;
static uint8_t joyHits = 0;
static uint8_t joyPending = 0;

static void calibrateJoystick() {
  long sx = 0, sy = 0;
  for (uint8_t i = 0; i < 16; i++) {
    sx += analogRead(PIN_JOY_X);
    sy += analogRead(PIN_JOY_Y);
    delay(2);
  }
  joyCx = sx / 16;
  joyCy = sy / 16;
}

// Returns the bay (1 up, 2 right, 3 down, 4 left) the stick is flicked towards, or 0.
static uint8_t pollJoystick() {
  int dx = analogRead(PIN_JOY_X) - joyCx;
  int dy = analogRead(PIN_JOY_Y) - joyCy;
  if (JOY_SWAP_XY) {
    int tmp = dx;
    dx = dy;
    dy = tmp;
  }
  if (JOY_INVERT_X) dx = -dx;
  if (JOY_INVERT_Y) dy = -dy;  // after this, +dy = up
  int ax = abs(dx), ay = abs(dy);

  if (!joyArmed) {
    if (ax < JOY_REARM && ay < JOY_REARM) joyArmed = true;
    return 0;
  }
  if (ax < JOY_FIRE && ay < JOY_FIRE) {
    joyHits = 0;
    return 0;
  }
  uint8_t bay = ay >= ax ? (dy > 0 ? 1 : 3) : (dx > 0 ? 2 : 4);
  if (bay != joyPending) {
    joyPending = bay;
    joyHits = 0;
  }
  if (++joyHits < JOY_CONFIRM) return 0;
  joyArmed = false;
  joyHits = 0;
  return bay;
}

// ---------------------------------------------------------------- MPU -> MCU commands
// console_cmd runs on the Bridge thread; hand lines to loop() through a small mutexed ring.
static const uint8_t CMD_SLOTS = 8;
static const uint8_t CMD_LEN = 40;
static char cmdRing[CMD_SLOTS][CMD_LEN];
static uint8_t cmdHead = 0, cmdTail = 0;
static struct k_mutex cmdMutex;

static bool console_cmd(MsgPack::str_t line) {
  k_mutex_lock(&cmdMutex, K_FOREVER);
  uint8_t next = (cmdHead + 1) % CMD_SLOTS;
  bool ok = next != cmdTail;  // drop when full rather than block the bridge thread
  if (ok) {
    strncpy(cmdRing[cmdHead], line.c_str(), CMD_LEN - 1);
    cmdRing[cmdHead][CMD_LEN - 1] = '\0';
    cmdHead = next;
  }
  k_mutex_unlock(&cmdMutex);
  return ok;
}

static bool popCommand(char *out) {
  k_mutex_lock(&cmdMutex, K_FOREVER);
  bool ok = cmdTail != cmdHead;
  if (ok) {
    memcpy(out, cmdRing[cmdTail], CMD_LEN);
    cmdTail = (cmdTail + 1) % CMD_SLOTS;
  }
  k_mutex_unlock(&cmdMutex);
  return ok;
}

// ---------------------------------------------------------------- bay state
enum BayLed : uint8_t { LED_UNKNOWN, LED_NOMINAL, LED_STALLED, LED_QUEUED, LED_RESCUE,
                        LED_VALIDATING, LED_AUTO, LED_OOD };

static uint8_t bayLed[4] = {LED_UNKNOWN, LED_UNKNOWN, LED_UNKNOWN, LED_UNKNOWN};
static uint32_t lastCmdMs = 0;
static bool everHeardMpu = false;

static uint8_t parseLed(const char *tok) {
  if (!strcmp(tok, "NOMINAL")) return LED_NOMINAL;
  if (!strcmp(tok, "STALLED")) return LED_STALLED;
  if (!strcmp(tok, "QUEUED")) return LED_QUEUED;
  if (!strcmp(tok, "RESCUE")) return LED_RESCUE;
  if (!strcmp(tok, "VALIDATING")) return LED_VALIDATING;
  if (!strcmp(tok, "AUTO")) return LED_AUTO;
  if (!strcmp(tok, "OOD")) return LED_OOD;
  return LED_UNKNOWN;
}

static void handleCommand(const char *line) {
  lastCmdMs = millis();
  everHeardMpu = true;
  if (!strcmp(line, "CMD:RESET_RIG")) {
    rigLatched = false;  // re-armed: the next rising edge latches again
    return;
  }
  static const char prefix[] = "CMD:BAY_STATE:";
  if (!strncmp(line, prefix, sizeof(prefix) - 1)) {
    const char *p = line + sizeof(prefix) - 1;
    if (p[0] >= '1' && p[0] <= '4' && p[1] == ':') bayLed[p[0] - '1'] = parseLed(p + 2);
  }
}

// ---------------------------------------------------------------- bay LEDs (KY-011)
// Plain uint8_t colours: the Arduino builder hoists prototypes above any enum type.
static const uint8_t OFF = 0, RED = 1, GREEN = 2, YELLOW = 3;

static void writeBay(uint8_t b, uint8_t c) {
  digitalWrite(PIN_LED_RED[b], (c == RED || c == YELLOW) ? HIGH : LOW);
  digitalWrite(PIN_LED_GREEN[b], (c == GREEN || c == YELLOW) ? HIGH : LOW);
}

static uint8_t bayColour(uint8_t s, uint32_t t) {
  switch (s) {
    case LED_NOMINAL:    return GREEN;                         // solid green
    case LED_STALLED:    return (t / 125) % 2 ? RED : OFF;     // fast red blink
    case LED_QUEUED:     return YELLOW;                        // solid amber
    case LED_RESCUE:     return RED;                           // solid red: operator on it
    case LED_VALIDATING: return (t / 250) % 2 ? YELLOW : OFF;  // amber blink
    case LED_AUTO:       return (t / 500) % 2 ? GREEN : OFF;   // slow green blink
    case LED_OOD:        return (t / 250) % 2 ? RED : GREEN;   // red/green alternate
    default:             return OFF;                           // never told
  }
}

static void renderBayLeds(uint32_t t) {
  bool stop = estopActive();
  for (uint8_t b = 0; b < 4; b++) writeBay(b, stop ? RED : bayColour(bayLed[b], t));
}

static void lampTest() {  // boot self-test: each bay red, then green, so wiring is checkable
  for (uint8_t b = 0; b < 4; b++) {
    writeBay(b, RED);
    delay(150);
    writeBay(b, GREEN);
    delay(150);
    writeBay(b, OFF);
  }
}

// ---------------------------------------------------------------- buzzer pattern
static unsigned int buzzerFor(uint32_t t) {
  if (estopActive()) return (t / 150) % 2 ? 2500 : 0;  // E-stop alarm
  if ((int32_t)(chirpUntilMs - t) > 0) return chirpHz;  // one-shot feedback
  for (uint8_t b = 0; b < 4; b++) {
    if (bayLed[b] == LED_STALLED || bayLed[b] == LED_OOD) {
      return (t % 1000) < 80 ? 2000 : 0;  // short beep once a second: needs a human
    }
  }
  return 0;
}

// ---------------------------------------------------------------- LED matrix
static const uint8_t ROWS = 8, COLS = 13, TILE_W = 3, STATUS_COL = 12, MAX_LEVEL = 7;
static uint8_t frame[ROWS * COLS];

static inline void px(uint8_t row, uint8_t col, uint8_t level) {
  frame[row * COLS + col] = level > MAX_LEVEL ? MAX_LEVEL : level;
}

// Brightness 0..7 for one tile pixel of bay b at time t (ms).
static uint8_t tileLevel(uint8_t s, uint8_t row, uint8_t col, uint32_t t) {
  switch (s) {
    case LED_NOMINAL: {  // slow breathing pulse, 2 s period
      uint32_t ph = t % 2000;
      uint32_t tri = ph < 1000 ? ph : 2000 - ph;  // 0..1000
      return 1 + (uint8_t)(tri * 4 / 1000);       // 1..5
    }
    case LED_STALLED:  // fast blink, 4 Hz
      return (t / 125) % 2 ? MAX_LEVEL : 0;
    case LED_QUEUED:  // slower blink, 2 Hz, so queued bays read differently from stalled
      return (t / 250) % 2 ? MAX_LEVEL : 1;
    case LED_RESCUE:  // solid
      return MAX_LEVEL;
    case LED_VALIDATING: {  // horizontal bar sweeping down the tile
      uint8_t bar = (t / 60) % ROWS;
      return row == bar ? MAX_LEVEL : (row + 1 == bar || row == bar + 1 ? 2 : 0);
    }
    case LED_AUTO: {  // vertical bar sweeping across the tile: skill replaying
      uint8_t bar = (t / 120) % TILE_W;
      return col == bar ? MAX_LEVEL : 1;
    }
    case LED_OOD:  // alternating checkerboard: refused, needs a human
      return ((row + col + t / 250) % 2) ? MAX_LEVEL : 0;
    default:  // never told: faint dot pattern
      return (row % 2 == 0 && col == 1) ? 1 : 0;
  }
}

static void renderFrame(Arduino_LED_Matrix &matrix, uint32_t t) {
  for (uint8_t b = 0; b < 4; b++) {
    for (uint8_t r = 0; r < ROWS; r++) {
      for (uint8_t c = 0; c < TILE_W; c++) px(r, b * TILE_W + c, tileLevel(bayLed[b], r, c, t));
    }
    if (selectedBay == b + 1) {  // selected bay: bright underline on the bottom row
      for (uint8_t c = 0; c < TILE_W; c++) px(ROWS - 1, b * TILE_W + c, MAX_LEVEL);
    }
  }
  // Status column, two rows each: MPU link, clutch, fault rig, E-stop.
  bool link = everHeardMpu && (t - lastCmdMs) < LINK_TIMEOUT_MS;
  uint8_t linkLevel = link ? 2 : ((t / 500) % 2 ? MAX_LEVEL : 0);
  uint8_t clutchLevel = clutchHeld() ? MAX_LEVEL : 0;
  uint8_t rigLevel = rigLatched ? MAX_LEVEL : 0;
  uint8_t estopLevel = estopActive() ? ((t / 100) % 2 ? MAX_LEVEL : 0) : 0;
  const uint8_t levels[4] = {linkLevel, clutchLevel, rigLevel, estopLevel};
  for (uint8_t i = 0; i < 4; i++) {
    px(2 * i, STATUS_COL, levels[i]);
    px(2 * i + 1, STATUS_COL, levels[i]);
  }
  matrix.draw(frame);
}

// ---------------------------------------------------------------- setup / loop
static Arduino_LED_Matrix matrix;
static uint32_t lastPollUs = 0, lastJoyMs = 0, lastHbMs = 0, lastFrameMs = 0;

void setup() {
  k_mutex_init(&cmdMutex);
  for (uint8_t b = 0; b < 4; b++) {
    pinMode(PIN_LED_RED[b], OUTPUT);
    pinMode(PIN_LED_GREEN[b], OUTPUT);
    writeBay(b, OFF);
  }
  pinMode(PIN_BUZZER, OUTPUT);
  digitalWrite(PIN_BUZZER, LOW);

  initInput(clutchIn, PIN_CLUTCH, BUTTON_ACTIVE_LOW);
  initInput(estopIn, PIN_ESTOP, BUTTON_ACTIVE_LOW);
  initInput(rigIn, PIN_FAULT_RIG, true);
  initInput(joySwIn, PIN_JOY_SW, true);
  analogReadResolution(10);
  calibrateJoystick();  // the stick must be untouched at boot

  matrix.begin();
  matrix.setGrayscaleBits(3);  // levels 0..7
  matrix.clear();
  lampTest();

  Bridge.begin();
  Bridge.provide("console_cmd", console_cmd);

  // Report the boot state so the host starts in sync (a rig closed at boot does not latch).
  sendBit("CLUTCH", clutchIn.stable);
  sendBit("ESTOP", estopActive());
  sendHeartbeat();
  chirp(3000, 60);
  lastPollUs = micros();
  lastJoyMs = lastHbMs = lastFrameMs = millis();
}

static void pollInputs() {
  if (updateInput(clutchIn)) sendBit("CLUTCH", clutchIn.stable);
  if (updateInput(estopIn)) sendBit("ESTOP", estopActive());
  if (updateInput(rigIn) && rigIn.stable && !rigLatched) {  // one-way rising edge
    rigLatched = true;
    sendLine("FAULT_RIG:LATCHED");
    chirp(1500, 300);
  }
  // Joystick click re-announces the selected bay (handy if the host missed the flick).
  if (updateInput(joySwIn) && joySwIn.stable && selectedBay) {
    sendBaySelect();
    chirp(3000, 30);
  }
}

void loop() {
  // loop() is paced by the core's loop hook (~1 ms when idle). Scan at most once per elapsed
  // millisecond: bursts of back-to-back reads would sample the same instant and defeat debounce.
  uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastPollUs) >= POLL_US) {
    lastPollUs = nowUs;
    pollInputs();
  }

  char cmd[CMD_LEN];
  while (popCommand(cmd)) handleCommand(cmd);

  uint32_t nowMs = millis();
  if (nowMs - lastJoyMs >= JOY_MS) {
    lastJoyMs = nowMs;
    uint8_t bay = pollJoystick();
    if (bay) {
      selectedBay = bay;
      sendBaySelect();
      chirp(3000, 30);
    }
#if JOY_DEBUG
    static uint8_t joyDbg = 0;
    if (++joyDbg >= 20) {  // 5 Hz
      joyDbg = 0;
      char buf[20];
      snprintf(buf, sizeof(buf), "JOY:%d:%d", analogRead(PIN_JOY_X), analogRead(PIN_JOY_Y));
      sendLine(buf);
    }
#endif
  }
  if (nowMs - lastHbMs >= HEARTBEAT_MS) {
    lastHbMs = nowMs;
    sendHeartbeat();
  }
  if (nowMs - lastFrameMs >= FRAME_MS) {
    lastFrameMs = nowMs;
    renderFrame(matrix, nowMs);
    renderBayLeds(nowMs);
    setBuzzer(buzzerFor(nowMs));
  }
}
