; Auto-generated. Do not edit!


(cl:in-package hybrid_simulation-msg)


;//! \htmlinclude ChangeLane.msg.html

(cl:defclass <ChangeLane> (roslisp-msg-protocol:ros-message)
  ((lane_change
    :reader lane_change
    :initarg :lane_change
    :type cl:fixnum
    :initform 0))
)

(cl:defclass ChangeLane (<ChangeLane>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <ChangeLane>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'ChangeLane)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_simulation-msg:<ChangeLane> is deprecated: use hybrid_simulation-msg:ChangeLane instead.")))

(cl:ensure-generic-function 'lane_change-val :lambda-list '(m))
(cl:defmethod lane_change-val ((m <ChangeLane>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:lane_change-val is deprecated.  Use hybrid_simulation-msg:lane_change instead.")
  (lane_change m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <ChangeLane>) ostream)
  "Serializes a message object of type '<ChangeLane>"
  (cl:let* ((signed (cl:slot-value msg 'lane_change)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <ChangeLane>) istream)
  "Deserializes a message object of type '<ChangeLane>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'lane_change) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<ChangeLane>)))
  "Returns string type for a message object of type '<ChangeLane>"
  "hybrid_simulation/ChangeLane")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'ChangeLane)))
  "Returns string type for a message object of type 'ChangeLane"
  "hybrid_simulation/ChangeLane")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<ChangeLane>)))
  "Returns md5sum for a message object of type '<ChangeLane>"
  "21070bac28cd495dd1acc43133eea981")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'ChangeLane)))
  "Returns md5sum for a message object of type 'ChangeLane"
  "21070bac28cd495dd1acc43133eea981")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<ChangeLane>)))
  "Returns full string definition for message of type '<ChangeLane>"
  (cl:format cl:nil "# Message to control high level actions of the Ego-Vehicle~%~%~%# lane_change : Change lane (0 keep lane; 1 lane change right; 2 lane change left)~%~%int16 lane_change~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'ChangeLane)))
  "Returns full string definition for message of type 'ChangeLane"
  (cl:format cl:nil "# Message to control high level actions of the Ego-Vehicle~%~%~%# lane_change : Change lane (0 keep lane; 1 lane change right; 2 lane change left)~%~%int16 lane_change~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <ChangeLane>))
  (cl:+ 0
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <ChangeLane>))
  "Converts a ROS message object to a list"
  (cl:list 'ChangeLane
    (cl:cons ':lane_change (lane_change msg))
))
