;; Auto-generated. Do not edit!


(when (boundp 'hybrid_simulation::Observations)
  (if (not (find-package "HYBRID_SIMULATION"))
    (make-package "HYBRID_SIMULATION"))
  (shadow 'Observations (find-package "HYBRID_SIMULATION")))
(unless (find-package "HYBRID_SIMULATION::OBSERVATIONS")
  (make-package "HYBRID_SIMULATION::OBSERVATIONS"))

(in-package "ROS")
;;//! \htmlinclude Observations.msg.html


(defclass hybrid_simulation::Observations
  :super ros::object
  :slots (_front_left _front _front_right _center_left _center_right _rear_left _rear_right _back_left _back_right _lane _dist_goal ))

(defmethod hybrid_simulation::Observations
  (:init
   (&key
    ((:front_left __front_left) 0)
    ((:front __front) 0)
    ((:front_right __front_right) 0)
    ((:center_left __center_left) 0)
    ((:center_right __center_right) 0)
    ((:rear_left __rear_left) 0)
    ((:rear_right __rear_right) 0)
    ((:back_left __back_left) 0)
    ((:back_right __back_right) 0)
    ((:lane __lane) 0)
    ((:dist_goal __dist_goal) 0.0)
    )
   (send-super :init)
   (setq _front_left (round __front_left))
   (setq _front (round __front))
   (setq _front_right (round __front_right))
   (setq _center_left (round __center_left))
   (setq _center_right (round __center_right))
   (setq _rear_left (round __rear_left))
   (setq _rear_right (round __rear_right))
   (setq _back_left (round __back_left))
   (setq _back_right (round __back_right))
   (setq _lane (round __lane))
   (setq _dist_goal (float __dist_goal))
   self)
  (:front_left
   (&optional __front_left)
   (if __front_left (setq _front_left __front_left)) _front_left)
  (:front
   (&optional __front)
   (if __front (setq _front __front)) _front)
  (:front_right
   (&optional __front_right)
   (if __front_right (setq _front_right __front_right)) _front_right)
  (:center_left
   (&optional __center_left)
   (if __center_left (setq _center_left __center_left)) _center_left)
  (:center_right
   (&optional __center_right)
   (if __center_right (setq _center_right __center_right)) _center_right)
  (:rear_left
   (&optional __rear_left)
   (if __rear_left (setq _rear_left __rear_left)) _rear_left)
  (:rear_right
   (&optional __rear_right)
   (if __rear_right (setq _rear_right __rear_right)) _rear_right)
  (:back_left
   (&optional __back_left)
   (if __back_left (setq _back_left __back_left)) _back_left)
  (:back_right
   (&optional __back_right)
   (if __back_right (setq _back_right __back_right)) _back_right)
  (:lane
   (&optional __lane)
   (if __lane (setq _lane __lane)) _lane)
  (:dist_goal
   (&optional __dist_goal)
   (if __dist_goal (setq _dist_goal __dist_goal)) _dist_goal)
  (:serialization-length
   ()
   (+
    ;; int8 _front_left
    1
    ;; int8 _front
    1
    ;; int8 _front_right
    1
    ;; int8 _center_left
    1
    ;; int8 _center_right
    1
    ;; int8 _rear_left
    1
    ;; int8 _rear_right
    1
    ;; int8 _back_left
    1
    ;; int8 _back_right
    1
    ;; int8 _lane
    1
    ;; float32 _dist_goal
    4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; int8 _front_left
       (write-byte _front_left s)
     ;; int8 _front
       (write-byte _front s)
     ;; int8 _front_right
       (write-byte _front_right s)
     ;; int8 _center_left
       (write-byte _center_left s)
     ;; int8 _center_right
       (write-byte _center_right s)
     ;; int8 _rear_left
       (write-byte _rear_left s)
     ;; int8 _rear_right
       (write-byte _rear_right s)
     ;; int8 _back_left
       (write-byte _back_left s)
     ;; int8 _back_right
       (write-byte _back_right s)
     ;; int8 _lane
       (write-byte _lane s)
     ;; float32 _dist_goal
       (sys::poke _dist_goal (send s :buffer) (send s :count) :float) (incf (stream-count s) 4)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; int8 _front_left
     (setq _front_left (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _front_left 127) (setq _front_left (- _front_left 256)))
   ;; int8 _front
     (setq _front (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _front 127) (setq _front (- _front 256)))
   ;; int8 _front_right
     (setq _front_right (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _front_right 127) (setq _front_right (- _front_right 256)))
   ;; int8 _center_left
     (setq _center_left (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _center_left 127) (setq _center_left (- _center_left 256)))
   ;; int8 _center_right
     (setq _center_right (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _center_right 127) (setq _center_right (- _center_right 256)))
   ;; int8 _rear_left
     (setq _rear_left (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _rear_left 127) (setq _rear_left (- _rear_left 256)))
   ;; int8 _rear_right
     (setq _rear_right (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _rear_right 127) (setq _rear_right (- _rear_right 256)))
   ;; int8 _back_left
     (setq _back_left (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _back_left 127) (setq _back_left (- _back_left 256)))
   ;; int8 _back_right
     (setq _back_right (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _back_right 127) (setq _back_right (- _back_right 256)))
   ;; int8 _lane
     (setq _lane (sys::peek buf ptr- :char)) (incf ptr- 1)
     (if (> _lane 127) (setq _lane (- _lane 256)))
   ;; float32 _dist_goal
     (setq _dist_goal (sys::peek buf ptr- :float)) (incf ptr- 4)
   ;;
   self)
  )

(setf (get hybrid_simulation::Observations :md5sum-) "a9c83c991797fc3e633dc6b433db3a15")
(setf (get hybrid_simulation::Observations :datatype-) "hybrid_simulation/Observations")
(setf (get hybrid_simulation::Observations :definition-)
      "# Message of the observations for decision making

# Possible_speeds: -100 Free;  0 Static; 1 Slow; 2 Fast; 100 Blocked

int8 front_left
int8 front
int8 front_right
int8 center_left
int8 center_right
int8 rear_left
int8 rear_right
int8 back_left
int8 back_right
# lane: -1 right of goal;  0 goal lane; 1 Left of lane
int8 lane
# dist_goal: Distance (m) to end of road / exit / end lane
float32 dist_goal
")



(provide :hybrid_simulation/Observations "a9c83c991797fc3e633dc6b433db3a15")


