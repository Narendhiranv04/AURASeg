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

class Observations {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.front_left = null;
      this.front = null;
      this.front_right = null;
      this.center_left = null;
      this.center_right = null;
      this.rear_left = null;
      this.rear_right = null;
      this.back_left = null;
      this.back_right = null;
      this.lane = null;
      this.dist_goal = null;
    }
    else {
      if (initObj.hasOwnProperty('front_left')) {
        this.front_left = initObj.front_left
      }
      else {
        this.front_left = 0;
      }
      if (initObj.hasOwnProperty('front')) {
        this.front = initObj.front
      }
      else {
        this.front = 0;
      }
      if (initObj.hasOwnProperty('front_right')) {
        this.front_right = initObj.front_right
      }
      else {
        this.front_right = 0;
      }
      if (initObj.hasOwnProperty('center_left')) {
        this.center_left = initObj.center_left
      }
      else {
        this.center_left = 0;
      }
      if (initObj.hasOwnProperty('center_right')) {
        this.center_right = initObj.center_right
      }
      else {
        this.center_right = 0;
      }
      if (initObj.hasOwnProperty('rear_left')) {
        this.rear_left = initObj.rear_left
      }
      else {
        this.rear_left = 0;
      }
      if (initObj.hasOwnProperty('rear_right')) {
        this.rear_right = initObj.rear_right
      }
      else {
        this.rear_right = 0;
      }
      if (initObj.hasOwnProperty('back_left')) {
        this.back_left = initObj.back_left
      }
      else {
        this.back_left = 0;
      }
      if (initObj.hasOwnProperty('back_right')) {
        this.back_right = initObj.back_right
      }
      else {
        this.back_right = 0;
      }
      if (initObj.hasOwnProperty('lane')) {
        this.lane = initObj.lane
      }
      else {
        this.lane = 0;
      }
      if (initObj.hasOwnProperty('dist_goal')) {
        this.dist_goal = initObj.dist_goal
      }
      else {
        this.dist_goal = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type Observations
    // Serialize message field [front_left]
    bufferOffset = _serializer.int8(obj.front_left, buffer, bufferOffset);
    // Serialize message field [front]
    bufferOffset = _serializer.int8(obj.front, buffer, bufferOffset);
    // Serialize message field [front_right]
    bufferOffset = _serializer.int8(obj.front_right, buffer, bufferOffset);
    // Serialize message field [center_left]
    bufferOffset = _serializer.int8(obj.center_left, buffer, bufferOffset);
    // Serialize message field [center_right]
    bufferOffset = _serializer.int8(obj.center_right, buffer, bufferOffset);
    // Serialize message field [rear_left]
    bufferOffset = _serializer.int8(obj.rear_left, buffer, bufferOffset);
    // Serialize message field [rear_right]
    bufferOffset = _serializer.int8(obj.rear_right, buffer, bufferOffset);
    // Serialize message field [back_left]
    bufferOffset = _serializer.int8(obj.back_left, buffer, bufferOffset);
    // Serialize message field [back_right]
    bufferOffset = _serializer.int8(obj.back_right, buffer, bufferOffset);
    // Serialize message field [lane]
    bufferOffset = _serializer.int8(obj.lane, buffer, bufferOffset);
    // Serialize message field [dist_goal]
    bufferOffset = _serializer.float32(obj.dist_goal, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type Observations
    let len;
    let data = new Observations(null);
    // Deserialize message field [front_left]
    data.front_left = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [front]
    data.front = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [front_right]
    data.front_right = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [center_left]
    data.center_left = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [center_right]
    data.center_right = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [rear_left]
    data.rear_left = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [rear_right]
    data.rear_right = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [back_left]
    data.back_left = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [back_right]
    data.back_right = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [lane]
    data.lane = _deserializer.int8(buffer, bufferOffset);
    // Deserialize message field [dist_goal]
    data.dist_goal = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 14;
  }

  static datatype() {
    // Returns string type for a message object
    return 'hybrid_simulation/Observations';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'a9c83c991797fc3e633dc6b433db3a15';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Message of the observations for decision making
    
    # Possible_speeds: -100 Free;  0 Static; 1 Slow; 2 Fast; 100 Blocked
    
    int8 front_left
    int8 front
    int8 front_right
    int8 center_left
    int8 center_right
    int8 rear_left
    int8 rear_right
    int8 back_left
    int8 back_right
    # lane: -1 right of goal;  0 goal lane; 1 Left of lane
    int8 lane
    # dist_goal: Distance (m) to end of road / exit / end lane
    float32 dist_goal
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new Observations(null);
    if (msg.front_left !== undefined) {
      resolved.front_left = msg.front_left;
    }
    else {
      resolved.front_left = 0
    }

    if (msg.front !== undefined) {
      resolved.front = msg.front;
    }
    else {
      resolved.front = 0
    }

    if (msg.front_right !== undefined) {
      resolved.front_right = msg.front_right;
    }
    else {
      resolved.front_right = 0
    }

    if (msg.center_left !== undefined) {
      resolved.center_left = msg.center_left;
    }
    else {
      resolved.center_left = 0
    }

    if (msg.center_right !== undefined) {
      resolved.center_right = msg.center_right;
    }
    else {
      resolved.center_right = 0
    }

    if (msg.rear_left !== undefined) {
      resolved.rear_left = msg.rear_left;
    }
    else {
      resolved.rear_left = 0
    }

    if (msg.rear_right !== undefined) {
      resolved.rear_right = msg.rear_right;
    }
    else {
      resolved.rear_right = 0
    }

    if (msg.back_left !== undefined) {
      resolved.back_left = msg.back_left;
    }
    else {
      resolved.back_left = 0
    }

    if (msg.back_right !== undefined) {
      resolved.back_right = msg.back_right;
    }
    else {
      resolved.back_right = 0
    }

    if (msg.lane !== undefined) {
      resolved.lane = msg.lane;
    }
    else {
      resolved.lane = 0
    }

    if (msg.dist_goal !== undefined) {
      resolved.dist_goal = msg.dist_goal;
    }
    else {
      resolved.dist_goal = 0.0
    }

    return resolved;
    }
};

module.exports = Observations;
