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

class SetSpeed {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.desired_speed = null;
    }
    else {
      if (initObj.hasOwnProperty('desired_speed')) {
        this.desired_speed = initObj.desired_speed
      }
      else {
        this.desired_speed = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type SetSpeed
    // Serialize message field [desired_speed]
    bufferOffset = _serializer.float32(obj.desired_speed, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type SetSpeed
    let len;
    let data = new SetSpeed(null);
    // Deserialize message field [desired_speed]
    data.desired_speed = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'hybrid_simulation/SetSpeed';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4d5008c9d834e2c102355282755ead21';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Message to control high level actions of the Ego-Vehicle
    
    
    # desired_speed : Desired speed of the vehicle
    
    float32 desired_speed
    
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new SetSpeed(null);
    if (msg.desired_speed !== undefined) {
      resolved.desired_speed = msg.desired_speed;
    }
    else {
      resolved.desired_speed = 0.0
    }

    return resolved;
    }
};

module.exports = SetSpeed;
