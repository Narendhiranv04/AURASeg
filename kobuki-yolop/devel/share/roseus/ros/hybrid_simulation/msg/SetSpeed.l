;; Auto-generated. Do not edit!


(when (boundp 'hybrid_simulation::SetSpeed)
  (if (not (find-package "HYBRID_SIMULATION"))
    (make-package "HYBRID_SIMULATION"))
  (shadow 'SetSpeed (find-package "HYBRID_SIMULATION")))
(unless (find-package "HYBRID_SIMULATION::SETSPEED")
  (make-package "HYBRID_SIMULATION::SETSPEED"))

(in-package "ROS")
;;//! \htmlinclude SetSpeed.msg.html


(defclass hybrid_simulation::SetSpeed
  :super ros::object
  :slots (_desired_speed ))

(defmethod hybrid_simulation::SetSpeed
  (:init
   (&key
    ((:desired_speed __desired_speed) 0.0)
    )
   (send-super :init)
   (setq _desired_speed (float __desired_speed))
   self)
  (:desired_speed
   (&optional __desired_speed)
   (if __desired_speed (setq _desired_speed __desired_speed)) _desired_speed)
  (:serialization-length
   ()
   (+
    ;; float32 _desired_speed
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; float32 _desired_speed
       (sys::poke _desired_speed (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; float32 _desired_speed
     (setq _desired_speed (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get hybrid_simulation::SetSpeed :md5sum-) "4d5008c9d834e2c102355282755ead21")
(setf (get hybrid_simulation::SetSpeed :datatype-) "hybrid_simulation/SetSpeed")
(setf (get hybrid_simulation::SetSpeed :definition-)
      "# Message to control high level actions of the Ego-Vehicle


# desired_speed : Desired speed of the vehicle

float32 desired_speed



")



(provide :hybrid_simulation/SetSpeed "4d5008c9d834e2c102355282755ead21")


