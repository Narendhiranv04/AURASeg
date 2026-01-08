; Auto-generated. Do not edit!


(cl:in-package hybrid_simulation-msg)


;//! \htmlinclude VehicleStatus.msg.html

(cl:defclass <VehicleStatus> (roslisp-msg-protocol:ros-message)
  ((vehicle_id
    :reader vehicle_id
    :initarg :vehicle_id
    :type cl:string
    :initform "")
   (pos_x
    :reader pos_x
    :initarg :pos_x
    :type cl:float
    :initform 0.0)
   (pos_y
    :reader pos_y
    :initarg :pos_y
    :type cl:float
    :initform 0.0)
   (heading
    :reader heading
    :initarg :heading
    :type cl:float
    :initform 0.0)
   (velocity
    :reader velocity
    :initarg :velocity
    :type cl:float
    :initform 0.0)
   (max_vel
    :reader max_vel
    :initarg :max_vel
    :type cl:float
    :initform 0.0)
   (lane
    :reader lane
    :initarg :lane
    :type cl:fixnum
    :initform 0)
   (signals
    :reader signals
    :initarg :signals
    :type cl:fixnum
    :initform 0))
)

(cl:defclass VehicleStatus (<VehicleStatus>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <VehicleStatus>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'VehicleStatus)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_simulation-msg:<VehicleStatus> is deprecated: use hybrid_simulation-msg:VehicleStatus instead.")))

(cl:ensure-generic-function 'vehicle_id-val :lambda-list '(m))
(cl:defmethod vehicle_id-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:vehicle_id-val is deprecated.  Use hybrid_simulation-msg:vehicle_id instead.")
  (vehicle_id m))

(cl:ensure-generic-function 'pos_x-val :lambda-list '(m))
(cl:defmethod pos_x-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:pos_x-val is deprecated.  Use hybrid_simulation-msg:pos_x instead.")
  (pos_x m))

(cl:ensure-generic-function 'pos_y-val :lambda-list '(m))
(cl:defmethod pos_y-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:pos_y-val is deprecated.  Use hybrid_simulation-msg:pos_y instead.")
  (pos_y m))

(cl:ensure-generic-function 'heading-val :lambda-list '(m))
(cl:defmethod heading-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:heading-val is deprecated.  Use hybrid_simulation-msg:heading instead.")
  (heading m))

(cl:ensure-generic-function 'velocity-val :lambda-list '(m))
(cl:defmethod velocity-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:velocity-val is deprecated.  Use hybrid_simulation-msg:velocity instead.")
  (velocity m))

(cl:ensure-generic-function 'max_vel-val :lambda-list '(m))
(cl:defmethod max_vel-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:max_vel-val is deprecated.  Use hybrid_simulation-msg:max_vel instead.")
  (max_vel m))

(cl:ensure-generic-function 'lane-val :lambda-list '(m))
(cl:defmethod lane-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:lane-val is deprecated.  Use hybrid_simulation-msg:lane instead.")
  (lane m))

(cl:ensure-generic-function 'signals-val :lambda-list '(m))
(cl:defmethod signals-val ((m <VehicleStatus>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:signals-val is deprecated.  Use hybrid_simulation-msg:signals instead.")
  (signals m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <VehicleStatus>) ostream)
  "Serializes a message object of type '<VehicleStatus>"
  (cl:let ((__ros_str_len (cl:length (cl:slot-value msg 'vehicle_id))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) __ros_str_len) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) __ros_str_len) ostream))
  (cl:map cl:nil #'(cl:lambda (c) (cl:write-byte (cl:char-code c) ostream)) (cl:slot-value msg 'vehicle_id))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'pos_x))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'pos_y))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'heading))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'velocity))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'max_vel))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
  (cl:let* ((signed (cl:slot-value msg 'lane)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'signals)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 65536) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) unsigned) ostream)
    )
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <VehicleStatus>) istream)
  "Deserializes a message object of type '<VehicleStatus>"
    (cl:let ((__ros_str_len 0))
      (cl:setf (cl:ldb (cl:byte 8 0) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) __ros_str_len) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'vehicle_id) (cl:make-string __ros_str_len))
      (cl:dotimes (__ros_str_idx __ros_str_len msg)
        (cl:setf (cl:char (cl:slot-value msg 'vehicle_id) __ros_str_idx) (cl:code-char (cl:read-byte istream)))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'pos_x) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'pos_y) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'heading) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'velocity) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'max_vel) (roslisp-utils:decode-single-float-bits bits)))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'lane) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'signals) (cl:if (cl:< unsigned 32768) unsigned (cl:- unsigned 65536))))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<VehicleStatus>)))
  "Returns string type for a message object of type '<VehicleStatus>"
  "hybrid_simulation/VehicleStatus")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'VehicleStatus)))
  "Returns string type for a message object of type 'VehicleStatus"
  "hybrid_simulation/VehicleStatus")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<VehicleStatus>)))
  "Returns md5sum for a message object of type '<VehicleStatus>"
  "c81aa0791049124d486b5aa675fa06f6")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'VehicleStatus)))
  "Returns md5sum for a message object of type 'VehicleStatus"
  "c81aa0791049124d486b5aa675fa06f6")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<VehicleStatus>)))
  "Returns full string definition for message of type '<VehicleStatus>"
  (cl:format cl:nil "# Message to send information about Vehicles in the scene~%~%# id : The idenfification of the vehicle~%# pos_x : Vehicle x position~%# pos_y : Vehicle y position~%# heading : Vehicle heading (Yaw angle)~%# velocity : Linear velocity of the vehicle~%# max_vel : Maximum velocity~%~%string  vehicle_id~%float32 pos_x~%float32 pos_y~%float32 heading~%float32 velocity~%float32 max_vel~%int16 lane~%int16 signals~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'VehicleStatus)))
  "Returns full string definition for message of type 'VehicleStatus"
  (cl:format cl:nil "# Message to send information about Vehicles in the scene~%~%# id : The idenfification of the vehicle~%# pos_x : Vehicle x position~%# pos_y : Vehicle y position~%# heading : Vehicle heading (Yaw angle)~%# velocity : Linear velocity of the vehicle~%# max_vel : Maximum velocity~%~%string  vehicle_id~%float32 pos_x~%float32 pos_y~%float32 heading~%float32 velocity~%float32 max_vel~%int16 lane~%int16 signals~%~%~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <VehicleStatus>))
  (cl:+ 0
     4 (cl:length (cl:slot-value msg 'vehicle_id))
     4
     4
     4
     4
     4
     2
     2
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <VehicleStatus>))
  "Converts a ROS message object to a list"
  (cl:list 'VehicleStatus
    (cl:cons ':vehicle_id (vehicle_id msg))
    (cl:cons ':pos_x (pos_x msg))
    (cl:cons ':pos_y (pos_y msg))
    (cl:cons ':heading (heading msg))
    (cl:cons ':velocity (velocity msg))
    (cl:cons ':max_vel (max_vel msg))
    (cl:cons ':lane (lane msg))
    (cl:cons ':signals (signals msg))
))
