
"use strict";

let DigitalInputEvent = require('./DigitalInputEvent.js');
let ControllerInfo = require('./ControllerInfo.js');
let DockInfraRed = require('./DockInfraRed.js');
let KeyboardInput = require('./KeyboardInput.js');
let DigitalOutput = require('./DigitalOutput.js');
let ButtonEvent = require('./ButtonEvent.js');
let Sound = require('./Sound.js');
let RobotStateEvent = require('./RobotStateEvent.js');
let ExternalPower = require('./ExternalPower.js');
let VersionInfo = require('./VersionInfo.js');
let MotorPower = require('./MotorPower.js');
let CliffEvent = require('./CliffEvent.js');
let BumperEvent = require('./BumperEvent.js');
let SensorState = require('./SensorState.js');
let PowerSystemEvent = require('./PowerSystemEvent.js');
let Led = require('./Led.js');
let WheelDropEvent = require('./WheelDropEvent.js');
let ScanAngle = require('./ScanAngle.js');
let AutoDockingActionResult = require('./AutoDockingActionResult.js');
let AutoDockingFeedback = require('./AutoDockingFeedback.js');
let AutoDockingResult = require('./AutoDockingResult.js');
let AutoDockingGoal = require('./AutoDockingGoal.js');
let AutoDockingActionGoal = require('./AutoDockingActionGoal.js');
let AutoDockingActionFeedback = require('./AutoDockingActionFeedback.js');
let AutoDockingAction = require('./AutoDockingAction.js');

module.exports = {
  DigitalInputEvent: DigitalInputEvent,
  ControllerInfo: ControllerInfo,
  DockInfraRed: DockInfraRed,
  KeyboardInput: KeyboardInput,
  DigitalOutput: DigitalOutput,
  ButtonEvent: ButtonEvent,
  Sound: Sound,
  RobotStateEvent: RobotStateEvent,
  ExternalPower: ExternalPower,
  VersionInfo: VersionInfo,
  MotorPower: MotorPower,
  CliffEvent: CliffEvent,
  BumperEvent: BumperEvent,
  SensorState: SensorState,
  PowerSystemEvent: PowerSystemEvent,
  Led: Led,
  WheelDropEvent: WheelDropEvent,
  ScanAngle: ScanAngle,
  AutoDockingActionResult: AutoDockingActionResult,
  AutoDockingFeedback: AutoDockingFeedback,
  AutoDockingResult: AutoDockingResult,
  AutoDockingGoal: AutoDockingGoal,
  AutoDockingActionGoal: AutoDockingActionGoal,
  AutoDockingActionFeedback: AutoDockingActionFeedback,
  AutoDockingAction: AutoDockingAction,
};
