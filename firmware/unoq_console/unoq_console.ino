/*
  ATC operator console for the Arduino UNO Q STM32U585 MCU (T-131, spec 5.2).

  - Polls bay-select buttons 1-4, the hold-to-control dead-man's clutch, the E-stop and the
    judge's 3D-printed fault rig at 1 kHz (INPUT_PULLUP, 5 ms debounce).
  - Sends text lines to the Linux MPU through arduino-router: Bridge.notify("console_line", ..)
      BTN:BAY:<1-4>  CLUTCH:<0|1>  ESTOP:<0|1>  FAULT_RIG:LATCHED
      HB:<bay 0-4>:<clutch>:<estop>:<rig>   (20 Hz heartbeat)
  - Serves Bridge "console_cmd" for MPU -> MCU lines:
      CMD:BAY_STATE:<1-4>:<NOMINAL|STALLED|QUEUED|RESCUE|VALIDATING|AUTO|OOD>
      CMD:RESET_RIG
  - Drives the 8x13 LED matrix: one 3-column x 8-row tile per bay (columns 0-2, 3-5, 6-8,
    9-11) and a status column (12).

  Fault rig is strictly one-way: the first rising edge emits FAULT_RIG:LATCHED once, then every
  further toggle is ignored until CMD:RESET_RIG. Python side: src/teleop/unoq/mcu_bridge.py.

  Build: arduino-cli compile --fqbn arduino:zephyr:unoq firmware/unoq_console
*/

#include <Arduino_LED_Matrix.h>
#include <Arduino_RouterBridge.h>
#include <zephyr/kernel.h>

// ---------------------------------------------------------------- wiring (edit to match rig)
// Every input is a switch to GND with the internal pull-up, so LOW = closed.
static const uint8_t PIN_BAY[4] = {2, 3, 4, 5};  // momentary push buttons, closed = pressed
static const uint8_t PIN_CLUTCH = 6;             // hold-to-control, closed = held
static const uint8_t PIN_ESTOP = 7;              // see ESTOP_NORMALLY_CLOSED
static const uint8_t PIN_FAULT_RIG = 8;          // closed = flap raised ("rising edge")

// A normally-closed E-stop reads LOW when healthy, so a pressed button OR a cut wire reads
// HIGH = stopped. Set to false for a normally-open button (or nothing wired) on the bench.
#define ESTOP_NORMALLY_CLOSED true

// ---------------------------------------------------------------- timing
static const uint32_t POLL_US = 1000;       // 1 kHz input scan
static const uint8_t DEBOUNCE_MS = 5;       // level must be stable this long
static const uint32_t HEARTBEAT_MS = 50;    // 20 Hz
static const uint32_t FRAME_MS = 20;        // 50 Hz LED refresh
static const uint32_t LINK_TIMEOUT_MS = 2000;  // no MPU command for this long -> link lost

// ---------------------------------------------------------------- debounced inputs
struct Debounced {
  uint8_t pin;
  bool stable;   // debounced "closed" level
  bool raw;
  uint8_t count;
};

static Debounced bayBtn[4];
static Debounced clutchIn, estopIn, rigIn;

static bool readClosed(uint8_t pin) { return digitalRead(pin) == LOW; }

static void initInput(Debounced &d, uint8_t pin) {
  pinMode(pin, INPUT_PULLUP);
  d.pin = pin;
  d.raw = d.stable = readClosed(pin);
  d.count = 0;
}

// Returns true when the debounced level changed this tick.
static bool updateInput(Debounced &d) {
  bool now = readClosed(d.pin);
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

// ---------------------------------------------------------------- LED matrix
enum BayLed : uint8_t { LED_UNKNOWN, LED_NOMINAL, LED_STALLED, LED_QUEUED, LED_RESCUE,
                        LED_VALIDATING, LED_AUTO, LED_OOD };

static uint8_t bayLed[4] = {LED_UNKNOWN, LED_UNKNOWN, LED_UNKNOWN, LED_UNKNOWN};
static uint32_t lastCmdMs = 0;
static bool everHeardMpu = false;

static const uint8_t ROWS = 8, COLS = 13, TILE_W = 3, STATUS_COL = 12, MAX_LEVEL = 7;
static uint8_t frame[ROWS * COLS];

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

static void renderFrame(Arduino_LED_Matrix &matrix) {
  uint32_t t = millis();
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
  uint8_t clutchLevel = clutchIn.stable && !estopActive() ? MAX_LEVEL : 0;
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
static uint32_t lastPollUs = 0, lastHbMs = 0, lastFrameMs = 0;

void setup() {
  k_mutex_init(&cmdMutex);
  for (uint8_t i = 0; i < 4; i++) initInput(bayBtn[i], PIN_BAY[i]);
  initInput(clutchIn, PIN_CLUTCH);
  initInput(estopIn, PIN_ESTOP);
  initInput(rigIn, PIN_FAULT_RIG);

  matrix.begin();
  matrix.setGrayscaleBits(3);  // levels 0..7
  matrix.clear();

  Bridge.begin();
  Bridge.provide("console_cmd", console_cmd);

  // Report the boot state so the host starts in sync (a rig closed at boot does not latch).
  sendBit("CLUTCH", clutchIn.stable);
  sendBit("ESTOP", estopActive());
  sendHeartbeat();
  lastPollUs = micros();
  lastHbMs = lastFrameMs = millis();
}

static void pollInputs() {
  for (uint8_t i = 0; i < 4; i++) {
    if (updateInput(bayBtn[i]) && bayBtn[i].stable) {  // press edge only
      selectedBay = i + 1;
      char buf[12];
      snprintf(buf, sizeof(buf), "BTN:BAY:%u", selectedBay);
      sendLine(buf);
    }
  }
  if (updateInput(clutchIn)) sendBit("CLUTCH", clutchIn.stable);
  if (updateInput(estopIn)) sendBit("ESTOP", estopActive());
  if (updateInput(rigIn) && rigIn.stable && !rigLatched) {  // one-way rising edge
    rigLatched = true;
    sendLine("FAULT_RIG:LATCHED");
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
  if (nowMs - lastHbMs >= HEARTBEAT_MS) {
    lastHbMs = nowMs;
    sendHeartbeat();
  }
  if (nowMs - lastFrameMs >= FRAME_MS) {
    lastFrameMs = nowMs;
    renderFrame(matrix);
  }
}
