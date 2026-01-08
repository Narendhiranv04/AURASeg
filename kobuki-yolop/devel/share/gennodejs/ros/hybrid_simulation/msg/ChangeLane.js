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

class ChangeLane {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.lane_change = null;
    }
    else {
      if (initObj.hasOwnProperty('lane_change')) {
        this.lane_change = initObj.lane_change
      }
      else {
        this.lane_change = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type ChangeLane
    // Serialize message field [lane_change]
    bufferOffset = _serializer.int16(obj.lane_change, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type ChangeLane
    let len;
    let data = new ChangeLane(null);
    // Deserialize message field [lane_change]
    data.lane_change = _deserializer.int16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 2;
  }

  static datatype() {
    // Returns string type for a message object
    return 'hybrid_simulation/ChangeLane';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '21070bac28cd495dd1acc43133eea981';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Message to control high level actions of the Ego-Vehicle
    
    
    # lane_change : Change lane (0 keep lane; 1 lane change right; 2 lane change left)
    
    int16 lane_change
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new ChangeLane(null);
    if (msg.lane_change !== undefined) {
      resolved.lane_change = msg.lane_change;
    }
    else {
      resolved.lane_change = 0
    }

    return resolved;
    }
};

module.exports = ChangeLane;
