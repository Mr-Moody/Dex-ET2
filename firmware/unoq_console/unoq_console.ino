/*
  ATC operator console for the Arduino UNO Q STM32U585 MCU (T-131, T-216, spec 5.2).

  Hardware (matches the console wiring diagram, all modules on the 3V3 / GND rails):
    D2   Clutch button  KY-004 (hold-to-control dead-man's clutch)
    D3   Alert buzzer   KY-006 passive (driven with tone())
    D4/5 Bay 1 LED      KY-011 R/G      D6/7 Bay 2 LED  KY-011 G/R
    D8/9 Bay 3 LED      KY-011 R/G      D10/11 Bay 4 LED KY-011 G/R
    D12  E-stop button  KY-004 (per-robot power cut, see below)
    D13  Judge fault rig switch to GND (optional; unwired = never latches)
    A0/A1 Bay-select joystick KY-023 VRx/VRy, A2 its SW (optional)
    Onboard 8x13 LED matrix: one 3-column x 8-row tile per bay + a status column.

  - Polls clutch, E-stop, joystick switch and the fault rig at 1 kHz (5 ms debounce); samples
    the joystick at 100 Hz. Each stick direction selects the bay it is mapped to (default up 1,
    right 2, down 3, left 4; recalibrate on the console, see "Joystick calibration").
  - Buttons: the level read at boot is "released", so either KY-004 polarity works. Do not
    hold the clutch or the E-stop while the board boots.
  - E-stop latches per robot: a press cuts power to the selected bay (its LED goes solid red
    and the clutch no longer drives it); a second press with that bay selected restores it.
    With no bay selected a press cuts every bay (or restores all when all are cut).
  - Sends text lines to the Linux MPU through arduino-router: Bridge.notify("console_line", ..)
      BOOT                                  once per MCU boot (host restores map and power)
      BTN:BAY:<1-4>  CLUTCH:<0|1>  ESTOP:<0|1>  FAULT_RIG:LATCHED
      POWER:<bay 1-4>:<0|1>                 bay powered (1) or cut by the E-stop (0)
      JOYMAP:<4 digits>                     bays for stick X+, X-, Y+, Y- (raw ADC directions)
      JOYCAL:START  JOYCAL:<1-4>  JOYCAL:DUP:<1-4>  JOYCAL:OK:<map>  JOYCAL:FAIL:<reason>
      HB:<bay 0-4>:<clutch>:<estop>:<rig>:<power-off mask 0-15>   (20 Hz heartbeat)
      JOY:<x>:<y>                           raw stick at 10 Hz while CMD:JOY_RAW:1
  - Serves Bridge "console_cmd" for MPU -> MCU lines:
      CMD:BAY_STATE:<1-4>:<NOMINAL|STALLED|QUEUED|RESCUE|VALIDATING|AUTO|OOD>
      CMD:RESET_RIG
      CMD:JOY_MAP:<4 digits>   CMD:JOY_CAL   CMD:JOY_CAL_ABORT   CMD:JOY_RAW:<0|1>
      CMD:POWER_MASK:<0-15>    (bit b-1 set = bay b cut; the host restores it after BOOT)
  - Mirrors each bay state on its bi-colour LED and the matrix; beeps when a bay needs a human
    (STALLED / OOD) and every few seconds while any robot's power is cut.

  Joystick calibration: send CMD:JOY_CAL (teleop unoq --calibrate-joystick) or hold the stick
  pressed for 2 s. Let go of the stick while all four LEDs are amber (the centre is captured),
  then push the stick towards where you want each bay when its LED blinks amber: Bay 1, 2, 3, 4.
  A direction already taken buzzes low; push another one. Done bays turn green.

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

#define BUZZER_ENABLED true

// ---------------------------------------------------------------- timing
static const uint32_t POLL_US = 1000;          // 1 kHz input scan
static const uint8_t DEBOUNCE_MS = 5;          // level must be stable this long
static const uint32_t JOY_MS = 10;             // 100 Hz joystick sampling
static const uint32_t HEARTBEAT_MS = 50;       // 20 Hz
static const uint32_t FRAME_MS = 20;           // 50 Hz LED / buzzer refresh
static const uint32_t LINK_TIMEOUT_MS = 2000;  // no MPU command for this long -> link lost
static const uint32_t CAL_HOLD_MS = 2000;      // stick press that starts calibration
static const uint32_t CAL_SETTLE_MS = 800;     // hands off before the centre is captured
static const uint32_t CAL_TIMEOUT_MS = 20000;  // no direction for this long aborts calibration

// Joystick thresholds on the 10-bit ADC, measured from the centre captured at boot.
static const int JOY_FIRE = 300;       // deflection that selects a bay (~60% of half travel)
static const int JOY_REARM = 120;      // stick must come back inside this before the next flick
static const uint8_t JOY_CONFIRM = 3;  // consecutive samples past JOY_FIRE (30 ms)

// ---------------------------------------------------------------- debounced inputs
struct Debounced {
  uint8_t pin;
  uint8_t idle;  // raw level read at boot = released
  bool stable;   // debounced "closed / pressed" level
  bool raw;
  uint8_t count;
};

static Debounced clutchIn, estopIn, rigIn, joySwIn;

static bool readClosed(const Debounced &d) { return digitalRead(d.pin) != d.idle; }

// idleLevel < 0 learns the released level from the pin itself (KY-004 boards differ).
static void initInput(Debounced &d, uint8_t pin, int idleLevel) {
  pinMode(pin, INPUT_PULLUP);
  delay(2);
  d.pin = pin;
  d.idle = idleLevel < 0 ? digitalRead(pin) : (uint8_t)idleLevel;
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

// ---------------------------------------------------------------- console state
static uint8_t selectedBay = 0;  // 0 = none yet
static bool rigLatched = false;
static uint8_t powerOffMask = 0;  // bit b-1 set = bay b's power cut by the E-stop

static bool bayPowered(uint8_t bay) { return bay >= 1 && bay <= 4 && !(powerOffMask >> (bay - 1) & 1); }

static bool clutchHeld() {
  return clutchIn.stable && !estopIn.stable && selectedBay && bayPowered(selectedBay);
}

static void sendLine(const char *line) { Bridge.notify("console_line", MsgPack::str_t(line)); }

static void sendBit(const char *prefix, bool v) {
  char buf[16];
  snprintf(buf, sizeof(buf), "%s:%d", prefix, v ? 1 : 0);
  sendLine(buf);
}

static void sendHeartbeat() {
  char buf[28];
  snprintf(buf, sizeof(buf), "HB:%u:%d:%d:%d:%u", selectedBay, clutchIn.stable ? 1 : 0,
           estopIn.stable ? 1 : 0, rigLatched ? 1 : 0, powerOffMask);
  sendLine(buf);
}

static void sendBaySelect() {
  char buf[12];
  snprintf(buf, sizeof(buf), "BTN:BAY:%u", selectedBay);
  sendLine(buf);
}

static void sendPower(uint8_t bay) {
  char buf[12];
  snprintf(buf, sizeof(buf), "POWER:%u:%d", bay, bayPowered(bay) ? 1 : 0);
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

// ---------------------------------------------------------------- E-stop (per-robot power)
static void setPowerMask(uint8_t mask) {
  mask &= 0x0F;
  uint8_t changed = mask ^ powerOffMask;
  powerOffMask = mask;
  for (uint8_t b = 1; b <= 4; b++) {
    if (changed >> (b - 1) & 1) sendPower(b);
  }
}

static void onEstopPressed() {
  uint8_t mask;
  if (selectedBay) {
    mask = powerOffMask ^ (1 << (selectedBay - 1));
  } else {
    mask = powerOffMask == 0x0F ? 0 : 0x0F;
  }
  bool cut = mask & ~powerOffMask;
  setPowerMask(mask);
  if (cut) {
    chirp(400, 600);  // long low tone: power cut
  } else {
    chirp(2600, 120);  // short high tone: power restored
  }
}

// ---------------------------------------------------------------- joystick
// Stick directions in raw ADC terms (before any mapping): 0 X+, 1 X-, 2 Y+, 3 Y-.
static const uint8_t DIR_NONE = 0xFF;
static int joyCx = 512, joyCy = 512;
static bool joyArmed = true;
static uint8_t joyHits = 0;
static uint8_t joyPending = DIR_NONE;
static bool joyRaw = false;

// Bay selected by each direction. Default for the plate's mounting: up 1, right 2, down 3,
// left 4 with higher VRx = right and lower VRy = up.
static uint8_t joyMap[4] = {2, 4, 3, 1};

static void sendJoyMap(const char *prefix) {
  char buf[24];
  snprintf(buf, sizeof(buf), "%s%u%u%u%u", prefix, joyMap[0], joyMap[1], joyMap[2], joyMap[3]);
  sendLine(buf);
}

// Accepts exactly four digits that are a permutation of 1..4.
static bool parseJoyMap(const char *s, uint8_t out[4]) {
  uint8_t seen = 0;
  for (uint8_t i = 0; i < 4; i++) {
    if (s[i] < '1' || s[i] > '4') return false;
    out[i] = s[i] - '0';
    seen |= 1 << (out[i] - 1);
  }
  return s[4] == '\0' && seen == 0x0F;
}

static void captureJoyCentre() {
  long sx = 0, sy = 0;
  for (uint8_t i = 0; i < 16; i++) {
    sx += analogRead(PIN_JOY_X);
    sy += analogRead(PIN_JOY_Y);
    delay(2);
  }
  joyCx = sx / 16;
  joyCy = sy / 16;
}

// Returns the raw direction the stick is flicked towards (once per flick), or DIR_NONE.
static uint8_t pollJoystick() {
  int dx = analogRead(PIN_JOY_X) - joyCx;
  int dy = analogRead(PIN_JOY_Y) - joyCy;
  // An off-centre stick has less travel on one side; deflection is scaled to the travel left
  // in the pushed direction, so JOY_FIRE / JOY_REARM are counts out of a 512-count half travel.
  int travelX = dx > 0 ? 1023 - joyCx : joyCx;
  int travelY = dy > 0 ? 1023 - joyCy : joyCy;
  int ax = (int)((long)abs(dx) * 512 / max(travelX, 1));
  int ay = (int)((long)abs(dy) * 512 / max(travelY, 1));

  if (!joyArmed) {
    if (ax < JOY_REARM && ay < JOY_REARM) joyArmed = true;
    return DIR_NONE;
  }
  if (ax < JOY_FIRE && ay < JOY_FIRE) {
    joyHits = 0;
    return DIR_NONE;
  }
  uint8_t dir = ax >= ay ? (dx > 0 ? 0 : 1) : (dy > 0 ? 2 : 3);
  if (dir != joyPending) {
    joyPending = dir;
    joyHits = 0;
  }
  if (++joyHits < JOY_CONFIRM) return DIR_NONE;
  joyArmed = false;
  joyHits = 0;
  return dir;
}

// ---------------------------------------------------------------- joystick calibration
// calStep: -1 idle, 0 hands off (centre capture pending), 1..4 waiting for that bay's push.
static int8_t calStep = -1;
static uint32_t calSinceMs = 0;
static uint8_t calDirs[4];
static uint32_t swPressedMs = 0;
static bool swHoldFired = false;

static void startCalibration() {
  calStep = 0;
  calSinceMs = millis();
  for (uint8_t i = 0; i < 4; i++) calDirs[i] = DIR_NONE;
  sendLine("JOYCAL:START");
  chirp(1800, 150);
}

static void abortCalibration(const char *reason) {
  if (calStep < 0) return;
  calStep = -1;
  joyArmed = false;  // the stick must come home before the next flick selects a bay
  char buf[32];
  snprintf(buf, sizeof(buf), "JOYCAL:FAIL:%s", reason);
  sendLine(buf);
  chirp(300, 400);
}

static void promptCalStep() {
  char buf[12];
  snprintf(buf, sizeof(buf), "JOYCAL:%d", calStep);
  sendLine(buf);
}

static void stepCalibration(uint32_t now) {
  if (calStep == 0) {
    if (now - calSinceMs < CAL_SETTLE_MS) return;
    captureJoyCentre();
    joyArmed = true;
    calStep = 1;
    calSinceMs = now;
    promptCalStep();
    return;
  }
  if (now - calSinceMs > CAL_TIMEOUT_MS) {
    abortCalibration("TIMEOUT");
    return;
  }
  uint8_t dir = pollJoystick();
  if (dir == DIR_NONE) return;
  calSinceMs = now;
  for (int8_t i = 0; i < calStep - 1; i++) {
    if (calDirs[i] == dir) {
      char buf[16];
      snprintf(buf, sizeof(buf), "JOYCAL:DUP:%d", calStep);
      sendLine(buf);
      chirp(300, 200);
      return;
    }
  }
  calDirs[calStep - 1] = dir;
  if (calStep < 4) {
    calStep++;
    promptCalStep();
    chirp(3000, 40);
    return;
  }
  for (uint8_t b = 0; b < 4; b++) joyMap[calDirs[b]] = b + 1;
  calStep = -1;
  sendJoyMap("JOYCAL:OK:");
  sendJoyMap("JOYMAP:");
  chirp(3500, 250);
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

static bool startsWith(const char *s, const char *prefix, const char **rest) {
  size_t n = strlen(prefix);
  if (strncmp(s, prefix, n)) return false;
  *rest = s + n;
  return true;
}

static void handleCommand(const char *line) {
  lastCmdMs = millis();
  everHeardMpu = true;
  const char *p;
  if (!strcmp(line, "CMD:RESET_RIG")) {
    rigLatched = false;  // re-armed: the next rising edge latches again
  } else if (startsWith(line, "CMD:BAY_STATE:", &p)) {
    if (p[0] >= '1' && p[0] <= '4' && p[1] == ':') bayLed[p[0] - '1'] = parseLed(p + 2);
  } else if (startsWith(line, "CMD:JOY_MAP:", &p)) {
    uint8_t m[4];
    if (parseJoyMap(p, m)) memcpy(joyMap, m, sizeof(joyMap));
    sendJoyMap("JOYMAP:");  // ack with the map in force (unchanged on a bad request)
  } else if (!strcmp(line, "CMD:JOY_CAL")) {
    startCalibration();
  } else if (!strcmp(line, "CMD:JOY_CAL_ABORT")) {
    abortCalibration("ABORTED");
  } else if (startsWith(line, "CMD:JOY_RAW:", &p)) {
    joyRaw = p[0] == '1';
  } else if (startsWith(line, "CMD:POWER_MASK:", &p)) {
    int mask = atoi(p);
    if (mask >= 0 && mask <= 15) setPowerMask((uint8_t)mask);
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
    case LED_RESCUE:     return (t / 500) % 2 ? RED : YELLOW;  // red/amber: operator on it
    case LED_VALIDATING: return (t / 250) % 2 ? YELLOW : OFF;  // amber blink
    case LED_AUTO:       return (t / 500) % 2 ? GREEN : OFF;   // slow green blink
    case LED_OOD:        return (t / 250) % 2 ? RED : GREEN;   // red/green alternate
    default:             return OFF;                           // never told
  }
}

static uint8_t calColour(uint8_t b, uint32_t t) {
  if (calStep == 0) return YELLOW;  // hands off the stick
  if (b + 1 < (uint8_t)calStep) return GREEN;
  if (b + 1 == (uint8_t)calStep) return (t / 150) % 2 ? YELLOW : OFF;
  return OFF;
}

static void renderBayLeds(uint32_t t) {
  for (uint8_t b = 0; b < 4; b++) {
    uint8_t c;
    if (calStep >= 0) {
      c = calColour(b, t);
    } else if (!bayPowered(b + 1)) {
      c = RED;  // power cut: solid red until the E-stop restores it
    } else {
      c = bayColour(bayLed[b], t);
    }
    writeBay(b, c);
  }
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
  if ((int32_t)(chirpUntilMs - t) > 0) return chirpHz;  // one-shot feedback
  if (calStep >= 0) return 0;
  for (uint8_t b = 0; b < 4; b++) {
    if (bayLed[b] == LED_STALLED || bayLed[b] == LED_OOD) {
      return (t % 1000) < 80 ? 2000 : 0;  // short beep once a second: needs a human
    }
  }
  if (powerOffMask) return (t % 4000) < 60 ? 600 : 0;  // low tick: a robot is powered off
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

static uint8_t calTileLevel(uint8_t b, uint32_t t) {
  if (calStep == 0) return 2;
  if (b + 1 < (uint8_t)calStep) return 3;
  if (b + 1 == (uint8_t)calStep) return (t / 150) % 2 ? MAX_LEVEL : 0;
  return 0;
}

static void renderFrame(Arduino_LED_Matrix &matrix, uint32_t t) {
  for (uint8_t b = 0; b < 4; b++) {
    bool off = !bayPowered(b + 1);
    for (uint8_t r = 0; r < ROWS; r++) {
      for (uint8_t c = 0; c < TILE_W; c++) {
        uint8_t level;
        if (calStep >= 0) {
          level = calTileLevel(b, t);
        } else if (off) {  // power cut: hollow box
          level = (r == 0 || r == ROWS - 1 || c == 0 || c == TILE_W - 1) ? MAX_LEVEL : 0;
        } else {
          level = tileLevel(bayLed[b], r, c, t);
        }
        px(r, b * TILE_W + c, level);
      }
    }
    if (calStep < 0 && !off && selectedBay == b + 1) {  // selected bay: bright underline
      for (uint8_t c = 0; c < TILE_W; c++) px(ROWS - 1, b * TILE_W + c, MAX_LEVEL);
    }
  }
  // Status column, two rows each: MPU link, clutch, fault rig, any robot powered off.
  bool link = everHeardMpu && (t - lastCmdMs) < LINK_TIMEOUT_MS;
  uint8_t linkLevel = link ? 2 : ((t / 500) % 2 ? MAX_LEVEL : 0);
  uint8_t clutchLevel = clutchHeld() ? MAX_LEVEL : 0;
  uint8_t rigLevel = rigLatched ? MAX_LEVEL : 0;
  uint8_t powerLevel = powerOffMask ? ((t / 100) % 2 ? MAX_LEVEL : 0) : 0;
  const uint8_t levels[4] = {linkLevel, clutchLevel, rigLevel, powerLevel};
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

  initInput(clutchIn, PIN_CLUTCH, -1);
  initInput(estopIn, PIN_ESTOP, -1);
  initInput(rigIn, PIN_FAULT_RIG, HIGH);
  initInput(joySwIn, PIN_JOY_SW, HIGH);
  analogReadResolution(10);
  captureJoyCentre();  // the stick must be untouched at boot

  matrix.begin();
  matrix.setGrayscaleBits(3);  // levels 0..7
  matrix.clear();
  lampTest();

  Bridge.begin();
  Bridge.provide("console_cmd", console_cmd);

  // Report the boot state so the host starts in sync (a rig closed at boot does not latch).
  sendLine("BOOT");
  sendJoyMap("JOYMAP:");
  sendBit("CLUTCH", clutchIn.stable);
  sendBit("ESTOP", estopIn.stable);
  sendHeartbeat();
  chirp(3000, 60);
  lastPollUs = micros();
  lastJoyMs = lastHbMs = lastFrameMs = millis();
}

static void pollInputs() {
  if (updateInput(clutchIn)) sendBit("CLUTCH", clutchIn.stable);
  if (updateInput(estopIn)) {
    sendBit("ESTOP", estopIn.stable);
    if (estopIn.stable) onEstopPressed();
  }
  if (updateInput(rigIn) && rigIn.stable && !rigLatched) {  // one-way rising edge
    rigLatched = true;
    sendLine("FAULT_RIG:LATCHED");
    chirp(1500, 300);
  }
  if (updateInput(joySwIn)) {
    if (joySwIn.stable) {
      swPressedMs = millis();
      swHoldFired = false;
      // A click re-announces the selected bay (handy if the host missed the flick).
      if (selectedBay && calStep < 0) {
        sendBaySelect();
        chirp(3000, 30);
      }
    }
  }
  if (joySwIn.stable && !swHoldFired && millis() - swPressedMs >= CAL_HOLD_MS) {
    swHoldFired = true;
    if (calStep < 0) startCalibration();
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
    if (calStep >= 0) {
      stepCalibration(nowMs);
    } else {
      uint8_t dir = pollJoystick();
      if (dir != DIR_NONE) {
        selectedBay = joyMap[dir];
        sendBaySelect();
        chirp(3000, 30);
      }
    }
    static uint8_t joyDbg = 0;
    if (joyRaw && ++joyDbg >= 10) {  // 10 Hz
      joyDbg = 0;
      char buf[20];
      snprintf(buf, sizeof(buf), "JOY:%d:%d", analogRead(PIN_JOY_X), analogRead(PIN_JOY_Y));
      sendLine(buf);
    }
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
