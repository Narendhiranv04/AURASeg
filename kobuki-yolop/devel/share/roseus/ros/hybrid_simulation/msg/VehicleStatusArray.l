;; Auto-generated. Do not edit!


(when (boundp 'hybrid_simulation::VehicleStatusArray)
  (if (not (find-package "HYBRID_SIMULATION"))
    (make-package "HYBRID_SIMULATION"))
  (shadow 'VehicleStatusArray (find-package "HYBRID_SIMULATION")))
(unless (find-package "HYBRID_SIMULATION::VEHICLESTATUSARRAY")
  (make-package "HYBRID_SIMULATION::VEHICLESTATUSARRAY"))

(in-package "ROS")
;;//! \htmlinclude VehicleStatusArray.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass hybrid_simulation::VehicleStatusArray
  :super ros::object
  :slots (_header _VehiclesDetected ))

(defmethod hybrid_simulation::VehicleStatusArray
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:VehiclesDetected __VehiclesDetected) (let (r) (dotimes (i 0) (push (instance hybrid_simulation::VehicleStatus :init) r)) r))
    )
   (send-super :init)
   (setq _header __header)
   (setq _VehiclesDetected __VehiclesDetected)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:VehiclesDetected
   (&rest __VehiclesDetected)
   (if (keywordp (car __VehiclesDetected))
       (send* _VehiclesDetected __VehiclesDetected)
     (progn
       (if __VehiclesDetected (setq _VehiclesDetected (car __VehiclesDetected)))
       _VehiclesDetected)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; hybrid_simulation/VehicleStatus[] _VehiclesDetected
    (apply #'+ (send-all _VehiclesDetected :serialization-length)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; hybrid_simulation/VehicleStatus[] _VehiclesDetected
     (write-long (length _VehiclesDetected) s)
     (dolist (elem _VehiclesDetected)
       (send elem :serialize s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; hybrid_simulation/VehicleStatus[] _VehiclesDetected
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _VehiclesDetected (let (r) (dotimes (i n) (push (instance hybrid_simulation::VehicleStatus :init) r)) r))
     (dolist (elem- _VehiclesDetected)
     (send elem- :deserialize buf ptr-) (incf ptr- (send elem- :serialization-length))
     ))
   ;;
   self)
  )

(setf (get hybrid_simulation::VehicleStatusArray :md5sum-) "919422845bd2de82c89aae1314b50aa7")
(setf (get hybrid_simulation::VehicleStatusArray :datatype-) "hybrid_simulation/VehicleStatusArray")
(setf (get hybrid_simulation::VehicleStatusArray :definition-)
      "# Array variable message of VehicleStatus
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


")



(provide :hybrid_simulation/VehicleStatusArray "919422845bd2de82c89aae1314b50aa7")


