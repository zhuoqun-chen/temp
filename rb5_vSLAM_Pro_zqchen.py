#!/usr/bin/env python
import rospy
import csv
import numpy as np
import math
import time
from numpy.linalg import multi_dot
from geometry_msgs.msg import Twist
from april_detection.msg import AprilTagDetectionArray
from tf.transformations import euler_from_quaternion
from rb5_visual_servo_control import PIDcontroller, genTwistMsg, coord

# Write CSV file
timestr = time.strftime("%Y%m%d-%H%M")
# timestr = "this-file"
# timestr = "20221110" + "18dian20"
fh = open('/root/cse276a/ws5/src/telemetry_data/'+timestr+'_path.csv', 'w')
writer = csv.writer(fh)

class EKF_vSLAM:
    def __init__(self, var_System_noise, var_Sensor_noise, sensor_Error):
        """
        Initialize pose and covariance. 
        
        Parameters
        ----------
        var_System_noise: list
            Variance of the vehicle model/system noise (size 2)
        var_Sensor_noise: list 
            Variance of the sensor noise for r and phi (size 2)
        sensor_Error: float 
            Sensor error measurement threshold
        """
        self.var_System_noise = var_System_noise 
        self.var_Sensor_noise = var_Sensor_noise
        self.sensor_Error = sensor_Error
        
        # Initialization
        self.mu = np.zeros((3, 1))                  # Pose of the vehicle and positions of the landmark. (M - number of landmarks), (3 + 2M, 1)
        
        # >>> zqchen HW5 >>>
        #change it to my init location please
        self.mu = np.array([[0.61,0.61,0.0]]).T      # For generated path
        # <<< zqchen HW5 <<<
        
        self.cov = np.zeros((3, 3))                 # Covariance matrix of the state model, which is the uncertainty in the pose/state estimate, (3 + 2M, 3 + 2M)
        self.observed = []                          # List that stores observed Apriltags' id
    
    def predict_EKF(self, twist):
        """
        EKF Prediction Step 
        
        Parameters
        ----------
        twist: numpy.ndarray (3 + 2M, 1)
            Control vector, which includes twist vector (i.e., linear velocities, and angular velocity)
        dt: float
            Timestep
            
        Return
        ----------
        self.mu: numpy.ndarray (3 + 2M, 1)
            Estimated pose of the vehicle and positions of the landmark. (M - number of landmarks), (3 + 2M, 1)
        self.cov: numpy.ndarray (3 + 2M, 3 + 2M)
            Estimated covariance matrix of the state model, which is the uncertainty in the pose/state estimate, (3 + 2M, 3 + 2M)
        """
        vx, vy, w = twist           # Get vehicle linear velocities and angular velocity
        
        # Get the length of the vehicle state vector.
        mu_len = len(self.mu)
        
        # Define the F matrix, which is the autonomous evloution
        F = np.eye(mu_len)
        # Define the G control matrix
        G = np.zeros((mu_len, mu_len))
        G[0, 0], G[1, 1], G[2, 2] = 1.0, 1.0, 1.0
        # Define the u control vector
        u = np.zeros((mu_len, 1))
        u[0,0], u[1,0], u[2,0] = vx, vy, w
        # Define Qt matrix, which is the uncertainty of the model/system noise
        self.Qt = np.zeros((mu_len, mu_len)) 
        self.Qt[0,0], self.Qt[1,1], self.Qt[2,2] = self.var_System_noise[0], self.var_System_noise[0], self.var_System_noise[1]
        
        # Estimate the state
        # self.mu = F @ self.mu + G @ u
        self.mu = multi_dot([F, self.mu]) + multi_dot([G, u])
        # Estimate the covariance
        # self.cov = F @ self.cov @ F.T + self.Qt
        self.cov = multi_dot([F, self.cov, F.T]) + self.Qt
        
        return self.mu, self.cov
        
    def update_EKF(self, landmarks):
        """
        EKF Update Step 
        
        Parameters
        ----------
        landmarks:
            Detected landmarks
            
        Return
        ----------
        self.mu: numpy.ndarray (3 + 2M, 1)
            Updated pose of the vehicle and positions of the landmark. (M - number of landmarks), (3 + 2M, 1)
        self.cov: numpy.ndarray (3 + 2M, 3 + 2M)
            Updated covariance matrix of the state model, which is the uncertainty in the pose/state estimate, (3 + 2M, 3 + 2M)
        """
        
        x, y, theta = self.mu[:3, 0]           # Get estimated vehicle pose
        
        # Get the length of the state vector.
        mu_len = len(self.mu)

        # Define Sensor Noise
        var_r, var_phi = self.var_Sensor_noise
        
        # tag_id, curr_r, curr_z, curr_x
        for posX_landmark, posY_landmark, tagID in landmarks:
            
            r = np.linalg.norm(np.array([posX_landmark, posY_landmark]))
            phi = math.atan2(posX_landmark, posY_landmark)
            
            
            # if tagID not in self.observed:
            #     self.observed.append(tagID)         # Append to the observed list
            #     j = self.observed.index(tagID)      # Get the index of the tagID from the observed list
            
            # Landmark position in world frame
            landmark_x = x + r * math.cos(phi + theta)          
            landmark_y = y + r * math.sin(phi + theta)

            # >>> HW5 >>>
            # If it is an unique tagID
            if tagID not in self.observed:
            # <<< HW5 <<<

                # Vertically stack to mew
                self.mu = np.vstack((self.mu, landmark_x, landmark_y))
                # Get the length of the vehicle state vector.
                mu_len = len(self.mu)
                # Update Covariance size
                self.cov = np.block([[self.cov, np.zeros((mu_len-2,2))],
                                     [np.zeros((2, mu_len-2)), np.diag(np.array([1e6, 1e6]))]])
                
            # j = self.observed.index(tagID)      # Get the index of the tagID from the observed list  
            # idx = 3 + 2 * j                     # Determine the index of the tagID for the state vector


                self.observed.append(tagID)         # Append to the observed list
                j = len(self.observed) - 1          # Get the index of the tagID from the observed list
                idx = 3 + 2 * j                     # Determine the index of the tagID for the state vector

            # If it is not an unique tagID
            elif tagID in self.observed:
                
                # Indices with all the same tagID
                j_list = np.where(np.array(self.observed) == tagID)[0].tolist()
                # error list
                error_list = []
                
                # Calculate the error for each tagID
                for possible_j in j_list:
                    possible_idx = 3 + 2 * possible_j
                    abs_err = np.array([[abs(self.mu[possible_idx][0] - landmark_x)],[abs(self.mu[possible_idx+1][0] - landmark_y)]])
                    lse = np.linalg.norm(abs_err)
                    # If error is less than set threshold
                    if lse <= self.sensor_Error:
                        error_list.append((lse, possible_j))
                
                # If error_list is not empty
                if error_list:
                    error_list.sort()                   # Sort error list
                    j = error_list[0][1]                # Get the j index with lowest error
                    idx = 3 + 2 * j                     # Determine the index of the tagID for the state vector
                    
                else:
                    
                    # Vertically stack to mew
                    self.mu = np.vstack((self.mu, landmark_x, landmark_y))
                    # Get the length of the vehicle state vector.
                    mu_len = len(self.mu)
                    # Update Covariance size
                    self.cov = np.block([[self.cov, np.zeros((mu_len-2,2))],
                                        [np.zeros((2, mu_len-2)), np.diag(np.array([1e6, 1e6]))]])
                    
                    self.observed.append(tagID)         # Append to the observed list
                    j = len(self.observed) - 1          # Get the index of the tagID from the observed list
                    idx = 3 + 2 * j                     # Determine the index of the tagID for the state vector

            # Determine the distance between landmark position and vehicle position [2, 1]
            delta = np.array([[self.mu[idx][0] - x], [self.mu[idx+1][0] - y]])
            # Determine q (scalar)
            q = multi_dot([delta.T, delta])[0][0]

            z_tilde = np.array([[np.sqrt(q)], [math.atan2(delta[1][0], delta[0][0]) - theta]])
            
            # Create Fxj matrix that map from 2 to 2M + 3
            Fxj = np.zeros((5, mu_len))
            Fxj[:3,:3] = np.eye(3)
            Fxj[3, idx], Fxj[4, idx+1] = 1, 1

            # Define Jacobian Matrix for the Sensor Measurement
            J = 1 / q * np.array([
                [-np.sqrt(q)*delta[0][0], -np.sqrt(q)*delta[1][0], 0, np.sqrt(q)*delta[0][0], np.sqrt(q)*delta[1][0]],
                [delta[1][0], -delta[0][0], -q, -delta[1][0], delta[0][0]]                
                ])
            
            # Calculate H, which is the measurement prediction p(z |s ) ie a prediction of where features 
            # in the world are in the sensory frame [2, 3+2M]
            H = multi_dot([J, Fxj])

            # Define the sensor noise matrix Rt [2, 2]
            Rt = np.diag(np.array([var_r, var_phi]))

            # Calculate the Kalman Gain, K [3+2M, 2]
            K = multi_dot([self.cov, H.T, np.linalg.inv(multi_dot([H, self.cov, H.T]) + Rt)])
        
            # Define sensor measurement, z
            z = np.array([[r], [phi]])
            
            # Calculate measurement error [2L, 1]
            delta_z =  z - z_tilde
            delta_z[1][0] = (delta_z[1][0] + np.pi) % (2 * np.pi) - np.pi
        
            # Update mu
            self.mu = self.mu + multi_dot([K, delta_z])
            # Update cov
            self.cov = multi_dot([(np.eye(mu_len) - multi_dot([K, H])), self.cov])
            
        return self.mu, self.cov


if __name__ == "__main__":

    # Initialize node
    rospy.init_node("vSLAM")
    # Intialize publisher
    pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
    
    # Define callback for subscriber
    apriltag_detected = False                # April Detected Boolean
    landmarks_Info = []                      # msg.detections
    def apriltag_callback(msg):
        global apriltag_detected
        global landmarks_Info
        if len(msg.detections) == 0:
            apriltag_detected = False
        else:
            apriltag_detected = True
            landmarks_Info = msg.detections
    # Initialize subscriber
    rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, apriltag_callback)

    # Square Path
    # waypoint = np.array([[0.0,0.0,0.0],
    #                      [1.0,0.0,0.0],
    #                      [1.0,0.0,np.pi/2],
    #                      [1.0,1.0,np.pi/2],
    #                      [1.0,1.0,np.pi],
    #                       [0.0,1.0,np.pi],
    #                       [0.0,0.0,-np.pi/2]])
    
    # Octagon Path
    # waypoint = np.array([[0.0, 0.0, 0.0],
    #                      [0.61, 0.0, 0.0],
    #                      [1.22, 0.61, np.pi/2],
    #                      [1.22, 1.22, np.pi/2],
    #                      [0.61, 1.83, np.pi],
    #                      [0.0, 1.83, np.pi],
    #                      [-0.61, 1.22, -np.pi/2],
    #                      [-0.61, 0.61, -np.pi/2],
    #                      [0.0, 0.0, 0.0]])

    # waypoint = np.array([[0.0, 0.0, 0.0],
    #                      [0.5, 0.0, 0.0],
    #                     #  [0.8, 0.0, 0.0],
    #                     #  [0.5, 0.0, 0.0],
    #                     #  [0.8, 0.0, 0.0]
    #                     [0.8, 0.0, 0.0],
    #                     # [0.8, 0.0, np.pi/4],
    #                     # [0.8, 0.0, 0.0],
    #                     [0.8, 0.5, 0.0],
    #                     [0,0,0]
    #                     ])

    # square at Home
    unit = 0.5
    waypoint = np.array([[0.0, 0.0, 0.0],

                               [unit*1.0, unit*0.0, 0.0],
                               [unit*1.0, unit*0.0, np.pi/2],
                               [unit*1.0, unit*1.0, np.pi/2],
                               [unit*1.0, unit*2.0, np.pi/2],
                               [unit*1.0, unit*2.0, np.pi],
                               [unit*0.0, unit*2.0, np.pi],
                               [unit*0.0, unit*1.0, np.pi/2],
                               [unit*0.0, unit*0.0, np.pi/2],
                               [unit*0.0, unit*0.0, 0.0]
                              ])
    small = 0.9
    waypoint = np.array([[0.0, 0.0, 0.0],

                               [unit*1.0, unit*0.0, 0.0],
                               [unit*1.0, unit*0.0, np.pi/2],
                               [unit*1.0, unit*1.0, np.pi/2],
                               [unit*1.0, unit*2.0, np.pi/2+np.pi/4],
                            #    [unit*1.0, unit*2.0, np.pi],
                               [unit*0.0+0.1, unit*2.0, np.pi],
                               [unit*0.0+0.1, unit*1.0, np.pi+np.pi/4],
                               [unit*0.0+0.1, unit*0.0, np.pi+np.pi/2],
                               [unit*0.0+0.1, unit*0.0, 0.0]
                              ])
    
    # waypoint = np.array([[0.0, 0.0, 0.0],

    #                      [1, 0.0, 0.0],

    #                     [1.2, 0.0, 0.0],

    #                     [1.2, 0.8, 0.0],
    #                     [0,0,0]
    #                     ])

    waypoint = np.array([[0.0, 0.0, 0.0],

                         [1, 0.0, 0.0],

                        [1, 1, 0.0],

                        # [1, 1, np.pi],

                        # [0, 1, np.pi],

                        [0,1,0],
                        [0,1,np.pi/2],
                        [0,0,np.pi/2],

                        # [0, 0, np.pi],


                        [0,0,0]
                        ])

    # lab experiments - square experiment
    unit = 1*0.8/0.8
    waypoint = np.array([[0.0, 0.0, 0.0],

                                # move to (1, 0, 0)
                                [unit*0.1, unit*0.0, 0.0],
                                [unit*0.2, unit*0.0, 0.0],
                                [unit*0.3, unit*0.0, 0.0],
                                [unit*0.4, unit*0.0, 0.0],
                                [unit*0.5, unit*0.0, 0.0],
                                [unit*0.6, unit*0.1 *(1/5), np.pi/2 *(1/5)],
                                [unit*0.7, unit*0.1 *(2/5), np.pi/2 *(2/5)],
                                [unit*0.8, unit*0.1 *(3/5), np.pi/2 *(3/5)],
                                [unit*0.9, unit*0.1 *(4/5), np.pi/2 *(4/5)],

                                [unit*1.0, unit*0.1 *(5/5), np.pi/2 *(5/5)],

                                [unit*1.0, unit*0.2, np.pi/2 *(5/5)],
                                [unit*1.0, unit*0.3, np.pi/2 *(5/5)],
                                [unit*1.0, unit*0.4, np.pi/2 *(5/5)],
                                [unit*1.0, unit*0.5, np.pi/2 *(5/5)],

                                [unit*1.0, unit*0.6, np.pi/2 *(5/5) + np.pi/2*(1/5)],
                                [unit*1.0, unit*0.7, np.pi/2 *(5/5) + np.pi/2*(2/5)],
                                [unit*1.0, unit*0.8, np.pi/2 *(5/5) + np.pi/2*(3/5)],
                                [unit*1.0, unit*0.9, np.pi/2 *(5/5) + np.pi/2*(4/5)],
                                [unit*1.0, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],

                                [unit*1.0, unit*0.6, np.pi/2 *(5/5) + np.pi/2*(1/5)],
                                [unit*1.0, unit*0.7, np.pi/2 *(5/5) + np.pi/2*(2/5)],
                                [unit*1.0, unit*0.8, np.pi/2 *(5/5) + np.pi/2*(3/5)],
                                [unit*1.0, unit*0.9, np.pi/2 *(5/5) + np.pi/2*(4/5)],
                                [unit*1.0, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],

                                [unit*1.0-0.1*1, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*2, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*3, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*4, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*5, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],

                                [unit*1.0-0.1*6, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*7, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*8, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*9, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*1.0-0.1*10, unit*1.0, np.pi/2 *(5/5) + np.pi/2*(5/5)],


                                [unit*0, unit*1.0-0.1*1, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*2, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*3, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*4, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*5, np.pi/2 *(5/5) + np.pi/2*(5/5)],

                                [unit*0, unit*1.0-0.1*6, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*7, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*8, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*9, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                [unit*0, unit*1.0-0.1*10, np.pi/2 *(5/5) + np.pi/2*(5/5)],
                                

                            #    [unit*1.0, unit*0.0, 0.0],
                            #    [unit*1.5, unit*0.5, np.pi/2],
                            #    [unit*1.0, unit*1, np.pi/2],
                            #    [unit*1.0, unit*1, np.pi]
                            #    [unit*1.0, unit*1.0, np.pi]
                               
                            #    [unit*1.0, unit*2.0, np.pi/2],

                            #    [unit*1.0, unit*2.0, np.pi],
                            #    [unit*0.0, unit*2.0, np.pi],
                            #    [unit*0.0, unit*1.0, np.pi/2],
                            #    [unit*0.0, unit*0.0, np.pi/2],
                            #    [unit*0.0, unit*0.0, 0.0]
                              ])

    # lab experiments - octagen experiment
    waypoint = np.array([[0.0, 0.0, 0.0],

                        [1, 0.0, 0.0],

                    [1, 1, 0.0],

                    # [1, 1, np.pi],

                    # [0, 1, np.pi],

                    [0,1,0],
                    [0,1,np.pi/2],
                    [0,0,np.pi/2],

                    # [0, 0, np.pi],


                    [0,0,0]
                    ])

    ######### >>> HW4 >>> #########
    waypoints = np.array([
                    [0.5, 0.5, 0],
                    [1.0, 0.5, 0]
                    # [0.9, 0.9],
                    # [0.9, 2.1],
                    # [1.3, 2.5],
                    # [2.5, 2.5]
                ])
    ######### <<< HW4 <<< #########

    # >>> zqchen HW5 >>>
    # Generated Voronoi Path
    waypoint = np.array([[0.61,0.61,0.0],
                         [2.20,0.60,0.0],
                         [2.40,0.80,0.0],
                         [2.40,2.40,np.pi/2]
                         ])

    # First generate paths, then replace these values

    waypoint = np.array(
                       [[0.5,  0.5, 0],
                        [2.5,  0.5, 0],
                        [2.5,  1. , -np.pi],
                        [0.5,  1. , -np.pi],
                        [0.5,  1.5, 0],
                        [2.5,  1.5, 0],
                        [2.5,  2. , -np.pi],
                        [0.5,  2. , -np.pi],
                        [0.5,  2.5, 0],
                        [2.5,  2.5, 0]]
                         )
    # <<< zqchen HW5 <<<

    # init pid controller
    scale = 1.0
    #pid = PIDcontroller(0.03*scale, 0.002*scale, 0.00001*scale)
    #pid = PIDcontroller(0.02*scale, ccc0.005*scale, 0.00001*scale)
    
    # pid = PIDcontroller(0.02*scale, 0.003*scale, 0.00005*scale)

    # suitable 1
    # pid = PIDcontroller(0.02+0.002-0.002,0.002,0.0003, offset=0.001/2)
    
    # suuitable 2 perfect by now
    # pid = PIDcontroller(0.02, 0.002, 0.05)

    # suitable 3, bug is can't pull the origin
    # pid = PIDcontroller(0.02, 0.002, 0.05, offset=0.001)

    # pid = PIDcontroller(0.02, 0.002+0.002, 0.005-0.002, offset=0.001-0.001) # perfect at Home 
    pid = PIDcontroller(0.06, 0.002+0.002, 0.005-0.002-0.002, offset=0)
    pid = PIDcontroller(0.1, 0.002+0.002, 0.005-0.002-0.002, offset=0)


    # >>> zqchen HW5 >>>
    # This is the params on yu's robo, replace
    pid = PIDcontroller(0.04*scale, 0.0005*scale, 0.00005*scale)
    # <<< zqchen HW5 >>>


    # init ekf vslam
    # ekf_vSLAM = EKF_vSLAM(var_System_noise=[1e-6, 0.3], var_Sensor_noise=[1e-6, 3.05e-8])

    # >>> HW5 >>>
    ekf_vSLAM = EKF_vSLAM(var_System_noise=[0.1, 0.01], var_Sensor_noise=[0.01, 0.01], sensor_Error=0.43)
    # <<< HW5 <<<

    #ekf_vSLAM = EKF_vSLAM(var_System_noise=[1, 1], var_Sensor_noise=[1, 1])

    # init current state
    # current_state = np.array([1/np.sqrt(2),0.0,0.0])
    current_state = np.array([0.0, 0.0, 0.0])

    ### >>> HW4 >>> ###
    current_state = np.array([0.5, 0.5, 0.0])
    ### <<< HW4 <<< ###

    # >>> zqchen HW5 >>>
    # yu's param, replace please
    current_state = np.array([0.61,0.61,0.0])
    # <<< zqchen HW5 <<<

    covariance = np.zeros((3,3))
    
    # Initialize telemetry data acquisition
    t0 = time.time()
    time_counter = 0.0
    data = [time_counter] + current_state.tolist()
    writer.writerow(data)
    writer.writerow(covariance.flatten().tolist())

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough, 
    # the current way point will be updated with a new point.
    for wp in waypoint:
        
        print("move to way point", wp)
        # set wp as the target point
        pid.setTarget(wp)
        # calculate the current twist (delta x, delta y, and delta theta)
        update_value = pid.update(current_state)
        vehicle_twist = coord(update_value, current_state)
        # publish the twist
        pub_twist.publish(genTwistMsg(vehicle_twist))
        time.sleep(0.05)
        
        # Predict EKF
        joint_state, _ = ekf_vSLAM.predict_EKF(update_value)
        
        if apriltag_detected:
            
            # Get landmark
            landmarks = []

            # >>> zqchen HW5 >>>
            # only for aligning with yu's new rb5_vSLAM changes
            # Stores landmark ids
            landmark_ids = []
            # <<< zqchen HW5 <<<

            for landmark_info in landmarks_Info:
                tag_id = landmark_info.id
            # >>> zqchen HW5 >>>
            # only for aligning with yu's new rb5_vSLAM changes
            # Only accept unique landmarks to do localization
            if tag_id not in landmark_ids:
            # <<< zqchen HW5 <<<
                _, curr_r, _ = euler_from_quaternion(
                    [
                        landmark_info.pose.orientation.w,
                        landmark_info.pose.orientation.x,
                        landmark_info.pose.orientation.y,
                        landmark_info.pose.orientation.z,
                    ])
                curr_pose = landmark_info.pose.position
                curr_x, curr_z = -curr_pose.x, curr_pose.z
                landmarks.append([curr_x, curr_z, tag_id])  
                # >>> zqchen HW5 >>>
                # only for aligning with yu's new rb5_vSLAM changes
                landmark_ids.append(tag_id)
                # <<< zqchen HW5 <<<
                     
            # Update EKF
            joint_state, _ = ekf_vSLAM.update_EKF(landmarks)
            
        # Update the current state
        current_state = np.array([joint_state[0,0],joint_state[1,0],joint_state[2,0]])

        # Record telemetry        
        t1 = time.time()
        if t1 - t0 >= 0.2:
            time_counter += t1 - t0
            data = [time_counter] + joint_state.reshape(-1).tolist() + ekf_vSLAM.observed
            writer.writerow(data)
            print("I'm writing something*******************************************")
            writer.writerow(ekf_vSLAM.cov.flatten().tolist())
            t0 = t1
            

        #while(np.linalg.norm(pid.getError(current_state, wp)) > 0.30): # check the error between current state and current way point
        # while(np.linalg.norm(pid.getError(current_state, wp)) > 0.15): # at Home
        # while(np.linalg.norm(pid.getError(current_state, wp)) > 0.05): # at Home
        
        # >>> HW5 >>>
        while(np.linalg.norm(pid.getError(current_state, wp)) > 0.10):  
        # <<< HW5 <<<

            # calculate the current twist
            update_value = pid.update(current_state)
            vehicle_twist = coord(update_value, current_state)
            # publish the twist
            pub_twist.publish(genTwistMsg(vehicle_twist))
            time.sleep(0.05)
            
            # Predict EKF
            joint_state, _ = ekf_vSLAM.predict_EKF(update_value)
            
            if apriltag_detected:
                
                # Get landmark
                landmarks = []
                for landmark_info in landmarks_Info:
                    tag_id = landmark_info.id
                    _, curr_r, _ = euler_from_quaternion(
                        [
                            landmark_info.pose.orientation.w,
                            landmark_info.pose.orientation.x,
                            landmark_info.pose.orientation.y,
                            landmark_info.pose.orientation.z,
                        ])
                    curr_pose = landmark_info.pose.position
                    curr_x, curr_z = -curr_pose.x, curr_pose.z
                    landmarks.append([curr_x, curr_z, tag_id])  
                        
                # Update EKF
                joint_state, _ = ekf_vSLAM.update_EKF(landmarks)
                
            # Update the current state
            current_state = np.array([joint_state[0,0],joint_state[1,0],joint_state[2,0]])

            # Record telemetry        
            t1 = time.time()
            if t1 - t0 >= 0.2:
                time_counter += t1 - t0
                data = [time_counter] + joint_state.reshape(-1).tolist() + ekf_vSLAM.observed
                writer.writerow(data)
                print("I'm writing something*******************************************")
                writer.writerow(ekf_vSLAM.cov.flatten().tolist())
                t0 = t1
        
        # time.sleep(1)
                    
    # stop the car and exit
    pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))
    # Close csv file
    fh.close()
