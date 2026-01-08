; Auto-generated. Do not edit!


(cl:in-package hybrid_simulation-msg)


;//! \htmlinclude VehicleStatusArray.msg.html

(cl:defclass <VehicleStatusArray> (roslisp-msg-protocol:ros-message)
  ((header
    :reader header
    :initarg :header
    :type std_msgs-msg:Header
    :initform (cl:make-instance 'std_msgs-msg:Header))
   (VehiclesDetected
    :reader VehiclesDetected
    :initarg :VehiclesDetected
    :type (cl:vector hybrid_simulation-msg:VehicleStatus)
   :initform (cl:make-array 0 :element-type 'hybrid_simulation-msg:VehicleStatus :initial-element (cl:make-instance 'hybrid_simulation-msg:VehicleStatus))))
)

(cl:defclass VehicleStatusArray (<VehicleStatusArray>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <VehicleStatusArray>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'VehicleStatusArray)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_simulation-msg:<VehicleStatusArray> is deprecated: use hybrid_simulation-msg:VehicleStatusArray instead.")))

(cl:ensure-generic-function 'header-val :lambda-list '(m))
(cl:defmethod header-val ((m <VehicleStatusArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:header-val is deprecated.  Use hybrid_simulation-msg:header instead.")
  (header m))

(cl:ensure-generic-function 'VehiclesDetected-val :lambda-list '(m))
(cl:defmethod VehiclesDetected-val ((m <VehicleStatusArray>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:VehiclesDetected-val is deprecated.  Use hybrid_simulation-msg:VehiclesDetected instead.")
  (VehiclesDetected m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <VehicleStatusArray>) ostream)
  "Serializes a message object of type '<VehicleStatusArray>"
  (roslisp-msg-protocol:serialize (cl:slot-value msg 'header) ostream)
  (cl:let ((__ros_arr_len (cl:length (cl:slot-value msg 'VehiclesDetected))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_arr_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_arr_len) ostream))
  (cl:map cl:nil #'(cl:lambda (ele) (roslisp-msg-protocol:serialize ele ostream))
   (cl:slot-value msg 'VehiclesDetected))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <VehicleStatusArray>) istream)
  "Deserializes a message object of type '<VehicleStatusArray>"
  (roslisp-msg-protocol:deserialize (cl:slot-value msg 'header) istream)
  (cl:let ((__ros_arr_len 0))
    (cl:setf (cl:ldb (cl:byte 8 0) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 8) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 16) __ros_arr_len) (cl:read-byte istream))
    (cl:setf (cl:ldb (cl:byte 8 24) __ros_arr_len) (cl:read-byte istream))
  (cl:setf (cl:slot-value msg 'VehiclesDetected) (cl:make-array __ros_arr_len))
  (cl:let ((vals (cl:slot-value msg 'VehiclesDetected)))
    (cl:dotimes (i __ros_arr_len)
    (cl:setf (cl:aref vals i) (cl:make-instance 'hybrid_simulation-msg:VehicleStatus))
  (roslisp-msg-protocol:deserialize (cl:aref vals i) istream))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<VehicleStatusArray>)))
  "Returns string type for a message object of type '<VehicleStatusArray>"
  "hybrid_simulation/VehicleStatusArray")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'VehicleStatusArray)))
  "Returns string type for a message object of type 'VehicleStatusArray"
  "hybrid_simulation/VehicleStatusArray")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<VehicleStatusArray>)))
  "Returns md5sum for a message object of type '<VehicleStatusArray>"
  "919422845bd2de82c89aae1314b50aa7")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'VehicleStatusArray)))
  "Returns md5sum for a message object of type 'VehicleStatusArray"
  "919422845bd2de82c89aae1314b50aa7")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<VehicleStatusArray>)))
  "Returns full string definition for message of type '<VehicleStatusArray>"
  (cl:format cl:nil "# Array variable message of VehicleStatus~%Header header~%VehicleStatus[] VehiclesDetected~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: hybrid_simulation/VehicleStatus~%# Message to send information about Vehicles in the scene~%~%# id : The idenfification of the vehicle~%# pos_x : Vehicle x position~%# pos_y : Vehicle y position~%# heading : Vehicle heading (Yaw angle)~%# velocity : Linear velocity of the vehicle~%# max_vel : Maximum velocity~%~%string  vehicle_id~%float32 pos_x~%float32 pos_y~%float32 heading~%float32 velocity~%float32 max_vel~%int16 lane~%int16 signals~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'VehicleStatusArray)))
  "Returns full string definition for message of type 'VehicleStatusArray"
  (cl:format cl:nil "# Array variable message of VehicleStatus~%Header header~%VehicleStatus[] VehiclesDetected~%================================================================================~%MSG: std_msgs/Header~%# Standard metadata for higher-level stamped data types.~%# This is generally used to communicate timestamped data ~%# in a particular coordinate frame.~%# ~%# sequence ID: consecutively increasing ID ~%uint32 seq~%#Two-integer timestamp that is expressed as:~%# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')~%# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')~%# time-handling sugar is provided by the client library~%time stamp~%#Frame this data is associated with~%string frame_id~%~%================================================================================~%MSG: hybrid_simulation/VehicleStatus~%# Message to send information about Vehicles in the scene~%~%# id : The idenfification of the vehicle~%# pos_x : Vehicle x position~%# pos_y : Vehicle y position~%# heading : Vehicle heading (Yaw angle)~%# velocity : Linear velocity of the vehicle~%# max_vel : Maximum velocity~%~%string  vehicle_id~%float32 pos_x~%float32 pos_y~%float32 heading~%float32 velocity~%float32 max_vel~%int16 lane~%int16 signals~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <VehicleStatusArray>))
  (cl:+ 0
     (roslisp-msg-protocol:serialization-length (cl:slot-value msg 'header))
     4 (cl:reduce #'cl:+ (cl:slot-value msg 'VehiclesDetected) :key #'(cl:lambda (ele) (cl:declare (cl:ignorable ele)) (cl:+ (roslisp-msg-protocol:serialization-length ele))))
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <VehicleStatusArray>))
  "Converts a ROS message object to a list"
  (cl:list 'VehicleStatusArray
    (cl:cons ':header (header msg))
    (cl:cons ':VehiclesDetected (VehiclesDetected msg))
))
