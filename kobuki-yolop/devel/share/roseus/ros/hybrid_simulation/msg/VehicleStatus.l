;; Auto-generated. Do not edit!


(when (boundp 'hybrid_simulation::VehicleStatus)
  (if (not (find-package "HYBRID_SIMULATION"))
    (make-package "HYBRID_SIMULATION"))
  (shadow 'VehicleStatus (find-package "HYBRID_SIMULATION")))
(unless (find-package "HYBRID_SIMULATION::VEHICLESTATUS")
  (make-package "HYBRID_SIMULATION::VEHICLESTATUS"))

(in-package "ROS")
;;//! \htmlinclude VehicleStatus.msg.html


(defclass hybrid_simulation::VehicleStatus
  :super ros::object
  :slots (_vehicle_id _pos_x _pos_y _heading _velocity _max_vel _lane _signals ))

(defmethod hybrid_simulation::VehicleStatus
  (:init
   (&key
    ((:vehicle_id __vehicle_id) "")
    ((:pos_x __pos_x) 0.0)
    ((:pos_y __pos_y) 0.0)
    ((:heading __heading) 0.0)
    ((:velocity __velocity) 0.0)
    ((:max_vel __max_vel) 0.0)
    ((:lane __lane) 0)
    ((:signals __signals) 0)
    )
   (send-super :init)
   (setq _vehicle_id (string __vehicle_id))
   (setq _pos_x (float __pos_x))
   (setq _pos_y (float __pos_y))
   (setq _heading (float __heading))
   (setq _velocity (float __velocity))
   (setq _max_vel (float __max_vel))
   (setq _lane (round __lane))
   (setq _signals (round __signals))
   self)
  (:vehicle_id
   (&optional __vehicle_id)
   (if __vehicle_id (setq _vehicle_id __vehicle_id)) _vehicle_id)
  (:pos_x
   (&optional __pos_x)
   (if __pos_x (setq _pos_x __pos_x)) _pos_x)
  (:pos_y
   (&optional __pos_y)
   (if __pos_y (setq _pos_y __pos_y)) _pos_y)
  (:heading
   (&optional __heading)
   (if __heading (setq _heading __heading)) _heading)
  (:velocity
   (&optional __velocity)
   (if __velocity (setq _velocity __velocity)) _velocity)
  (:max_vel
   (&optional __max_vel)
   (if __max_vel (setq _max_vel __max_vel)) _max_vel)
  (:lane
   (&optional __lane)
   (if __lane (setq _lane __lane)) _lane)
  (:signals
   (&optional __signals)
   (if __signals (setq _signals __signals)) _signals)
  (:serialization-length
   ()
   (+
    ;; string _vehicle_id
    4 (length _vehicle_id)
    ;; float32 _pos_x
    4
    ;; float32 _pos_y
    4
    ;; float32 _heading
    4
    ;; float32 _velocity
    4
    ;; float32 _max_vel
    4
    ;; int16 _lane
    2
    ;; int16 _signals
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _vehicle_id
       (write-long (length _vehicle_id) s) (princ _vehicle_id s)
     ;; float32 _pos_x
       (sys::poke _pos_x (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _pos_y
       (sys::poke _pos_y (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _heading
       (sys::poke _heading (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _velocity
       (sys::poke _velocity (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; float32 _max_vel
       (sys::poke _max_vel (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;; int16 _lane
       (write-word _lane s)
     ;; int16 _signals
       (write-word _signals s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _vehicle_id
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _vehicle_id (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float32 _pos_x
     (setq _pos_x (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _pos_y
     (setq _pos_y (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _heading
     (setq _heading (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _velocity
     (setq _velocity (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; float32 _max_vel
     (setq _max_vel (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;; int16 _lane
     (setq _lane (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;; int16 _signals
     (setq _signals (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get hybrid_simulation::VehicleStatus :md5sum-) "c81aa0791049124d486b5aa675fa06f6")
(setf (get hybrid_simulation::VehicleStatus :datatype-) "hybrid_simulation/VehicleStatus")
(setf (get hybrid_simulation::VehicleStatus :definition-)
      "# Message to send information about Vehicles in the scene

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



(provide :hybrid_simulation/VehicleStatus "c81aa0791049124d486b5aa675fa06f6")


