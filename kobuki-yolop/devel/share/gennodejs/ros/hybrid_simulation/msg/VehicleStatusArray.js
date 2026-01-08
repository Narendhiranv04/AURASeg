// Auto-generated. Do not edit!

// (in-package hybrid_simulation.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let VehicleStatus = require('./VehicleStatus.js');
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class VehicleStatusArray {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.header = null;
      this.VehiclesDetected = null;
    }
    else {
      if (initObj.hasOwnProperty('header')) {
        this.header = initObj.header
      }
      else {
        this.header = new std_msgs.msg.Header();
      }
      if (initObj.hasOwnProperty('VehiclesDetected')) {
        this.VehiclesDetected = initObj.VehiclesDetected
      }
      else {
        this.VehiclesDetected = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type VehicleStatusArray
    // Serialize message field [header]
    bufferOffset = std_msgs.msg.Header.serialize(obj.header, buffer, bufferOffset);
    // Serialize message field [VehiclesDetected]
    // Serialize the length for message field [VehiclesDetected]
    bufferOffset = _serializer.uint32(obj.VehiclesDetected.length, buffer, bufferOffset);
    obj.VehiclesDetected.forEach((val) => {
      bufferOffset = VehicleStatus.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type VehicleStatusArray
    let len;
    let data = new VehicleStatusArray(null);
    // Deserialize message field [header]
    data.header = std_msgs.msg.Header.deserialize(buffer, bufferOffset);
    // Deserialize message field [VehiclesDetected]
    // Deserialize array length for message field [VehiclesDetected]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.VehiclesDetected = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.VehiclesDetected[i] = VehicleStatus.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += std_msgs.msg.Header.getMessageSize(object.header);
    object.VehiclesDetected.forEach((val) => {
      length += VehicleStatus.getMessageSize(val);
    });
    return length + 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'hybrid_simulation/VehicleStatusArray';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '919422845bd2de82c89aae1314b50aa7';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    # Array variable message of VehicleStatus
    Header header
    VehicleStatus[] VehiclesDetected
    ================================================================================
    MSG: std_msgs/Header
    # Standard metadata for higher-level stamped data types.
    # This is generally used to communicate timestamped data 
    # in a particular coordinate frame.
    # 
    # sequence ID: consecutively increasing ID 
    uint32 seq
    #Two-integer timestamp that is expressed as:
    # * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
    # * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
    # time-handling sugar is provided by the client library
    time stamp
    #Frame this data is associated with
    string frame_id
    
    ================================================================================
    MSG: hybrid_simulation/VehicleStatus
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
    const resolved = new VehicleStatusArray(null);
    if (msg.header !== undefined) {
      resolved.header = std_msgs.msg.Header.Resolve(msg.header)
    }
    else {
      resolved.header = new std_msgs.msg.Header()
    }

    if (msg.VehiclesDetected !== undefined) {
      resolved.VehiclesDetected = new Array(msg.VehiclesDetected.length);
      for (let i = 0; i < resolved.VehiclesDetected.length; ++i) {
        resolved.VehiclesDetected[i] = VehicleStatus.Resolve(msg.VehiclesDetected[i]);
      }
    }
    else {
      resolved.VehiclesDetected = []
    }

    return resolved;
    }
};

module.exports = VehicleStatusArray;
