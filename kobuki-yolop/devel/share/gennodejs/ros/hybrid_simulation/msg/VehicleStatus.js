// Auto-generated. Do not edit!

// (in-package hybrid_simulation.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class VehicleStatus {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.vehicle_id = null;
      this.pos_x = null;
      this.pos_y = null;
      this.heading = null;
      this.velocity = null;
      this.max_vel = null;
      this.lane = null;
      this.signals = null;
    }
    else {
      if (initObj.hasOwnProperty('vehicle_id')) {
        this.vehicle_id = initObj.vehicle_id
      }
      else {
        this.vehicle_id = '';
      }
      if (initObj.hasOwnProperty('pos_x')) {
        this.pos_x = initObj.pos_x
      }
      else {
        this.pos_x = 0.0;
      }
      if (initObj.hasOwnProperty('pos_y')) {
        this.pos_y = initObj.pos_y
      }
      else {
        this.pos_y = 0.0;
      }
      if (initObj.hasOwnProperty('heading')) {
        this.heading = initObj.heading
      }
      else {
        this.heading = 0.0;
      }
      if (initObj.hasOwnProperty('velocity')) {
        this.velocity = initObj.velocity
      }
      else {
        this.velocity = 0.0;
      }
      if (initObj.hasOwnProperty('max_vel')) {
        this.max_vel = initObj.max_vel
      }
      else {
        this.max_vel = 0.0;
      }
      if (initObj.hasOwnProperty('lane')) {
        this.lane = initObj.lane
      }
      else {
        this.lane = 0;
      }
      if (initObj.hasOwnProperty('signals')) {
        this.signals = initObj.signals
      }
      else {
        this.signals = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type VehicleStatus
    // Serialize message field [vehicle_id]
    bufferOffset = _serializer.string(obj.vehicle_id, buffer, bufferOffset);
    // Serialize message field [pos_x]
    bufferOffset = _serializer.float32(obj.pos_x, buffer, bufferOffset);
    // Serialize message field [pos_y]
    bufferOffset = _serializer.float32(obj.pos_y, buffer, bufferOffset);
    // Serialize message field [heading]
    bufferOffset = _serializer.float32(obj.heading, buffer, bufferOffset);
    // Serialize message field [velocity]
    bufferOffset = _serializer.float32(obj.velocity, buffer, bufferOffset);
    // Serialize message field [max_vel]
    bufferOffset = _serializer.float32(obj.max_vel, buffer, bufferOffset);
    // Serialize message field [lane]
    bufferOffset = _serializer.int16(obj.lane, buffer, bufferOffset);
    // Serialize message field [signals]
    bufferOffset = _serializer.int16(obj.signals, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type VehicleStatus
    let len;
    let data = new VehicleStatus(null);
    // Deserialize message field [vehicle_id]
    data.vehicle_id = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [pos_x]
    data.pos_x = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [pos_y]
    data.pos_y = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [heading]
    data.heading = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [velocity]
    data.velocity = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [max_vel]
    data.max_vel = _deserializer.float32(buffer, bufferOffset);
    // Deserialize message field [lane]
    data.lane = _deserializer.int16(buffer, bufferOffset);
    // Deserialize message field [signals]
    data.signals = _deserializer.int16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += object.vehicle_id.length;
    return length + 28;
  }

  static datatype() {
    // Returns string type for a message object
    return 'hybrid_simulation/VehicleStatus';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'c81aa0791049124d486b5aa675fa06f6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Message to send information about Vehicles in the scene
    
    # id : The idenfification of the vehicle
    # pos_x : Vehicle x position
    # pos_y : Vehicle y position
    # heading : Vehicle heading (Yaw angle)
    # velocity : Linear velocity of the vehicle
    # max_vel : Maximum velocity
    
    string  vehicle_id
    float32 pos_x
    float32 pos_y
    float32 heading
    float32 velocity
    float32 max_vel
    int16 lane
    int16 signals
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new VehicleStatus(null);
    if (msg.vehicle_id !== undefined) {
      resolved.vehicle_id = msg.vehicle_id;
    }
    else {
      resolved.vehicle_id = ''
    }

    if (msg.pos_x !== undefined) {
      resolved.pos_x = msg.pos_x;
    }
    else {
      resolved.pos_x = 0.0
    }

    if (msg.pos_y !== undefined) {
      resolved.pos_y = msg.pos_y;
    }
    else {
      resolved.pos_y = 0.0
    }

    if (msg.heading !== undefined) {
      resolved.heading = msg.heading;
    }
    else {
      resolved.heading = 0.0
    }

    if (msg.velocity !== undefined) {
      resolved.velocity = msg.velocity;
    }
    else {
      resolved.velocity = 0.0
    }

    if (msg.max_vel !== undefined) {
      resolved.max_vel = msg.max_vel;
    }
    else {
      resolved.max_vel = 0.0
    }

    if (msg.lane !== undefined) {
      resolved.lane = msg.lane;
    }
    else {
      resolved.lane = 0
    }

    if (msg.signals !== undefined) {
      resolved.signals = msg.signals;
    }
    else {
      resolved.signals = 0
    }

    return resolved;
    }
};

module.exports = VehicleStatus;
