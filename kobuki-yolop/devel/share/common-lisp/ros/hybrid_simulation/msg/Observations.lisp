; Auto-generated. Do not edit!


(cl:in-package hybrid_simulation-msg)


;//! \htmlinclude Observations.msg.html

(cl:defclass <Observations> (roslisp-msg-protocol:ros-message)
  ((front_left
    :reader front_left
    :initarg :front_left
    :type cl:fixnum
    :initform 0)
   (front
    :reader front
    :initarg :front
    :type cl:fixnum
    :initform 0)
   (front_right
    :reader front_right
    :initarg :front_right
    :type cl:fixnum
    :initform 0)
   (center_left
    :reader center_left
    :initarg :center_left
    :type cl:fixnum
    :initform 0)
   (center_right
    :reader center_right
    :initarg :center_right
    :type cl:fixnum
    :initform 0)
   (rear_left
    :reader rear_left
    :initarg :rear_left
    :type cl:fixnum
    :initform 0)
   (rear_right
    :reader rear_right
    :initarg :rear_right
    :type cl:fixnum
    :initform 0)
   (back_left
    :reader back_left
    :initarg :back_left
    :type cl:fixnum
    :initform 0)
   (back_right
    :reader back_right
    :initarg :back_right
    :type cl:fixnum
    :initform 0)
   (lane
    :reader lane
    :initarg :lane
    :type cl:fixnum
    :initform 0)
   (dist_goal
    :reader dist_goal
    :initarg :dist_goal
    :type cl:float
    :initform 0.0))
)

(cl:defclass Observations (<Observations>)
  ())

(cl:defmethod cl:initialize-instance :after ((m <Observations>) cl:&rest args)
  (cl:declare (cl:ignorable args))
  (cl:unless (cl:typep m 'Observations)
    (roslisp-msg-protocol:msg-deprecation-warning "using old message class name hybrid_simulation-msg:<Observations> is deprecated: use hybrid_simulation-msg:Observations instead.")))

(cl:ensure-generic-function 'front_left-val :lambda-list '(m))
(cl:defmethod front_left-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:front_left-val is deprecated.  Use hybrid_simulation-msg:front_left instead.")
  (front_left m))

(cl:ensure-generic-function 'front-val :lambda-list '(m))
(cl:defmethod front-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:front-val is deprecated.  Use hybrid_simulation-msg:front instead.")
  (front m))

(cl:ensure-generic-function 'front_right-val :lambda-list '(m))
(cl:defmethod front_right-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:front_right-val is deprecated.  Use hybrid_simulation-msg:front_right instead.")
  (front_right m))

(cl:ensure-generic-function 'center_left-val :lambda-list '(m))
(cl:defmethod center_left-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:center_left-val is deprecated.  Use hybrid_simulation-msg:center_left instead.")
  (center_left m))

(cl:ensure-generic-function 'center_right-val :lambda-list '(m))
(cl:defmethod center_right-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:center_right-val is deprecated.  Use hybrid_simulation-msg:center_right instead.")
  (center_right m))

(cl:ensure-generic-function 'rear_left-val :lambda-list '(m))
(cl:defmethod rear_left-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:rear_left-val is deprecated.  Use hybrid_simulation-msg:rear_left instead.")
  (rear_left m))

(cl:ensure-generic-function 'rear_right-val :lambda-list '(m))
(cl:defmethod rear_right-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:rear_right-val is deprecated.  Use hybrid_simulation-msg:rear_right instead.")
  (rear_right m))

(cl:ensure-generic-function 'back_left-val :lambda-list '(m))
(cl:defmethod back_left-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:back_left-val is deprecated.  Use hybrid_simulation-msg:back_left instead.")
  (back_left m))

(cl:ensure-generic-function 'back_right-val :lambda-list '(m))
(cl:defmethod back_right-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:back_right-val is deprecated.  Use hybrid_simulation-msg:back_right instead.")
  (back_right m))

(cl:ensure-generic-function 'lane-val :lambda-list '(m))
(cl:defmethod lane-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:lane-val is deprecated.  Use hybrid_simulation-msg:lane instead.")
  (lane m))

(cl:ensure-generic-function 'dist_goal-val :lambda-list '(m))
(cl:defmethod dist_goal-val ((m <Observations>))
  (roslisp-msg-protocol:msg-deprecation-warning "Using old-style slot reader hybrid_simulation-msg:dist_goal-val is deprecated.  Use hybrid_simulation-msg:dist_goal instead.")
  (dist_goal m))
(cl:defmethod roslisp-msg-protocol:serialize ((msg <Observations>) ostream)
  "Serializes a message object of type '<Observations>"
  (cl:let* ((signed (cl:slot-value msg 'front_left)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'front)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'front_right)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'center_left)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'center_right)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'rear_left)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'rear_right)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'back_left)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'back_right)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let* ((signed (cl:slot-value msg 'lane)) (unsigned (cl:if (cl:< signed 0) (cl:+ signed 256) signed)))
    (cl:write-byte (cl:ldb (cl:byte 8 0) unsigned) ostream)
    )
  (cl:let ((bits (roslisp-utils:encode-single-float-bits (cl:slot-value msg 'dist_goal))))
    (cl:write-byte (cl:ldb (cl:byte 8 0) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 8) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 16) bits) ostream)
    (cl:write-byte (cl:ldb (cl:byte 8 24) bits) ostream))
)
(cl:defmethod roslisp-msg-protocol:deserialize ((msg <Observations>) istream)
  "Deserializes a message object of type '<Observations>"
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'front_left) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'front) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'front_right) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'center_left) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'center_right) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'rear_left) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'rear_right) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'back_left) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'back_right) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((unsigned 0))
      (cl:setf (cl:ldb (cl:byte 8 0) unsigned) (cl:read-byte istream))
      (cl:setf (cl:slot-value msg 'lane) (cl:if (cl:< unsigned 128) unsigned (cl:- unsigned 256))))
    (cl:let ((bits 0))
      (cl:setf (cl:ldb (cl:byte 8 0) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 8) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 16) bits) (cl:read-byte istream))
      (cl:setf (cl:ldb (cl:byte 8 24) bits) (cl:read-byte istream))
    (cl:setf (cl:slot-value msg 'dist_goal) (roslisp-utils:decode-single-float-bits bits)))
  msg
)
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql '<Observations>)))
  "Returns string type for a message object of type '<Observations>"
  "hybrid_simulation/Observations")
(cl:defmethod roslisp-msg-protocol:ros-datatype ((msg (cl:eql 'Observations)))
  "Returns string type for a message object of type 'Observations"
  "hybrid_simulation/Observations")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql '<Observations>)))
  "Returns md5sum for a message object of type '<Observations>"
  "a9c83c991797fc3e633dc6b433db3a15")
(cl:defmethod roslisp-msg-protocol:md5sum ((type (cl:eql 'Observations)))
  "Returns md5sum for a message object of type 'Observations"
  "a9c83c991797fc3e633dc6b433db3a15")
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql '<Observations>)))
  "Returns full string definition for message of type '<Observations>"
  (cl:format cl:nil "# Message of the observations for decision making~%~%# Possible_speeds: -100 Free;  0 Static; 1 Slow; 2 Fast; 100 Blocked~%~%int8 front_left~%int8 front~%int8 front_right~%int8 center_left~%int8 center_right~%int8 rear_left~%int8 rear_right~%int8 back_left~%int8 back_right~%# lane: -1 right of goal;  0 goal lane; 1 Left of lane~%int8 lane~%# dist_goal: Distance (m) to end of road / exit / end lane~%float32 dist_goal~%~%"))
(cl:defmethod roslisp-msg-protocol:message-definition ((type (cl:eql 'Observations)))
  "Returns full string definition for message of type 'Observations"
  (cl:format cl:nil "# Message of the observations for decision making~%~%# Possible_speeds: -100 Free;  0 Static; 1 Slow; 2 Fast; 100 Blocked~%~%int8 front_left~%int8 front~%int8 front_right~%int8 center_left~%int8 center_right~%int8 rear_left~%int8 rear_right~%int8 back_left~%int8 back_right~%# lane: -1 right of goal;  0 goal lane; 1 Left of lane~%int8 lane~%# dist_goal: Distance (m) to end of road / exit / end lane~%float32 dist_goal~%~%"))
(cl:defmethod roslisp-msg-protocol:serialization-length ((msg <Observations>))
  (cl:+ 0
     1
     1
     1
     1
     1
     1
     1
     1
     1
     1
     4
))
(cl:defmethod roslisp-msg-protocol:ros-message-to-list ((msg <Observations>))
  "Converts a ROS message object to a list"
  (cl:list 'Observations
    (cl:cons ':front_left (front_left msg))
    (cl:cons ':front (front msg))
    (cl:cons ':front_right (front_right msg))
    (cl:cons ':center_left (center_left msg))
    (cl:cons ':center_right (center_right msg))
    (cl:cons ':rear_left (rear_left msg))
    (cl:cons ':rear_right (rear_right msg))
    (cl:cons ':back_left (back_left msg))
    (cl:cons ':back_right (back_right msg))
    (cl:cons ':lane (lane msg))
    (cl:cons ':dist_goal (dist_goal msg))
))
