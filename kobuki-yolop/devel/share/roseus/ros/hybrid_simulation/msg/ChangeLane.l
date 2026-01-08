;; Auto-generated. Do not edit!


(when (boundp 'hybrid_simulation::ChangeLane)
  (if (not (find-package "HYBRID_SIMULATION"))
    (make-package "HYBRID_SIMULATION"))
  (shadow 'ChangeLane (find-package "HYBRID_SIMULATION")))
(unless (find-package "HYBRID_SIMULATION::CHANGELANE")
  (make-package "HYBRID_SIMULATION::CHANGELANE"))

(in-package "ROS")
;;//! \htmlinclude ChangeLane.msg.html


(defclass hybrid_simulation::ChangeLane
  :super ros::object
  :slots (_lane_change ))

(defmethod hybrid_simulation::ChangeLane
  (:init
   (&key
    ((:lane_change __lane_change) 0)
    )
   (send-super :init)
   (setq _lane_change (round __lane_change))
   self)
  (:lane_change
   (&optional __lane_change)
   (if __lane_change (setq _lane_change __lane_change)) _lane_change)
  (:serialization-length
   ()
   (+
    ;; int16 _lane_change
    2
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int16 _lane_change
       (write-word _lane_change s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int16 _lane_change
     (setq _lane_change (sys::peek buf ptr- :short)) (incf ptr- 2)
   ;;
   self)
  )

(setf (get hybrid_simulation::ChangeLane :md5sum-) "21070bac28cd495dd1acc43133eea981")
(setf (get hybrid_simulation::ChangeLane :datatype-) "hybrid_simulation/ChangeLane")
(setf (get hybrid_simulation::ChangeLane :definition-)
      "# Message to control high level actions of the Ego-Vehicle


# lane_change : Change lane (0 keep lane; 1 lane change right; 2 lane change left)

int16 lane_change
")



(provide :hybrid_simulation/ChangeLane "21070bac28cd495dd1acc43133eea981")


