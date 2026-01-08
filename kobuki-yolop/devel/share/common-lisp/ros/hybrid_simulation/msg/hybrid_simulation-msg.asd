
(cl:in-package :asdf)

(defsystem "hybrid_simulation-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "ChangeLane" :depends-on ("_package_ChangeLane"))
    (:file "_package_ChangeLane" :depends-on ("_package"))
    (:file "Observations" :depends-on ("_package_Observations"))
    (:file "_package_Observations" :depends-on ("_package"))
    (:file "SetSpeed" :depends-on ("_package_SetSpeed"))
    (:file "_package_SetSpeed" :depends-on ("_package"))
    (:file "VehicleStatus" :depends-on ("_package_VehicleStatus"))
    (:file "_package_VehicleStatus" :depends-on ("_package"))
    (:file "VehicleStatusArray" :depends-on ("_package_VehicleStatusArray"))
    (:file "_package_VehicleStatusArray" :depends-on ("_package"))
  ))