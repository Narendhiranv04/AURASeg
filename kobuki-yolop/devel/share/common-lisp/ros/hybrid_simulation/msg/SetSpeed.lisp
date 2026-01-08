; Auto-generated. Do not edit!


(cl:in-package hybrid_simulation-msg)


;//! \htmlinclude SetSpeed.msg.html

(cl:defclass <SetSpeed> (roslisp-msg-protocol:ros-message)
  ((desired_speed
    :reader desired_speed
    :initarg :desired_speed
    :type cl:float
    :initform 0.0))
)

(cl:defclass SetSpeed (<SetSpeed>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <SetSpeed>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'SetSpeed)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_simulation-msg:<SetSpeed> is deprecated: use hybrid_simulation-msg:SetSpeed instead.")))

(cl:ensure-generic-function 'desired_speed-val :lambda-list '(m))
(cl:defmethod desired_speed-val ((m <SetSpeed>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:desired_speed-val is deprecated.  Use hybrid_simulation-msg:desired_speed instead.")
  (desired_speed m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <SetSpeed>) ostream)
  "Serializes a message object of type '<SetSpeed>"
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'desired_speed))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <SetSpeed>) istream)
  "Deserializes a message object of type '<SetSpeed>"
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'desired_speed) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<SetSpeed>)))
  "Returns string type for a message object of type '<SetSpeed>"
  "hybrid_simulation/SetSpeed")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'SetSpeed)))
  "Returns string type for a message object of type 'SetSpeed"
  "hybrid_simulation/SetSpeed")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<SetSpeed>)))
  "Returns md5sum for a message object of type '<SetSpeed>"
  "4d5008c9d834e2c102355282755ead21")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'SetSpeed)))
  "Returns md5sum for a message object of type 'SetSpeed"
  "4d5008c9d834e2c102355282755ead21")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<SetSpeed>)))
  "Returns full string definition for message of type '<SetSpeed>"
  (cl:format cl:nil "# Message to control high level actions of the Ego-Vehicle~%~%~%# desired_speed : Desired speed of the vehicle~%~%float32 desired_speed~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'SetSpeed)))
  "Returns full string definition for message of type 'SetSpeed"
  (cl:format cl:nil "# Message to control high level actions of the Ego-Vehicle~%~%~%# desired_speed : Desired speed of the vehicle~%~%float32 desired_speed~%~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <SetSpeed>))
  (cl:+ 0
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <SetSpeed>))
  "Converts a ROS message object to a list"
  (cl:list 'SetSpeed
    (cl:cons ':desired_speed (desired_speed msg))
))
