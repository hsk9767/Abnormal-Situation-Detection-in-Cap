import numpy as np

class Find_assault():
    def __init__(self):
        self.find_driver_count = 0
        self.eyes_nose_center_avg = 0
        self.eyes_nose_dist_avg = 0
        self.eyes_nose_dist_mul = 1
        self.find_driver_count_limit = 500

    def get_eyes_nose_dist_center(self, r_eye, l_eye, nose):
        dist = abs(r_eye[0] - nose[0]) + abs(r_eye[1] - nose[1]) + abs(l_eye[0] - nose[0]) + abs(l_eye[1] - nose[1])
        return self.eyes_nose_dist_mul * dist, (np.array(r_eye) + np.array(l_eye) + np.array(nose))/3

    def find_driver(self, current_poses, img_shape):
        ##img bbox 중 가장 오른쪽 x 좌표들만을 다 모아 놓는 것.
        tr_x_list = [ (pose.bbox[0] + pose.bbox[2]) for pose in current_poses ]
        tr_x_arr = np.array([x for x in tr_x_list if x > 2*img_shape[0]/3])
        if tr_x_arr.size > 0:
            most_right_index = np.argmax(tr_x_arr)
        else: ##driver 찾는 과정에서 아무런 오른쪽 부분에 아무런 사람이 존재하지 않으면 다시 찾기 시작해야 함.
            self.find_driver_count = 0
            return False ## return 값은 driver_find 플래그에 대한 bool 값

        ##평균과 거리 구하기
        eyes_nose_dist, eyes_nose_center = self.get_eyes_nose_dist_center(current_poses[most_right_index].keypoints[-4],
                                                                          current_poses[most_right_index].keypoints[-3],
                                                                          current_poses[most_right_index].keypoints[0] )

            ##첫 번째라면 지수이동평균을 측정 값을 주고 첫 번째가 아니라면 지수이동평균에 수식 대입
        if self.find_driver_count == 0:
            self.eyes_nose_dist_avg = eyes_nose_dist
            self.eyes_nose_center_avg = eyes_nose_center
            self.find_driver_count += 1
            return False
        elif self.find_driver_count > 0:
            self.eyes_nose_dist_avg = 0.99 * eyes_nose_dist + (1 - 0.99) * self.eyes_nose_dist_avg
            dist_from_center_avg = np.sum(np.abs(eyes_nose_center - self.eyes_nose_center_avg))
            if dist_from_center_avg > self.eyes_nose_dist_avg:
                self.find_driver_count = 0
                return False
            elif self.find_driver_count < self.find_driver_count_limit:
                self.find_driver_count += 1
                self.eyes_nose_center_avg = 0.99 * eyes_nose_center + (1 - 0.99) * self.eyes_nose_center_avg
                current_poses[most_right_index].id = 'DRIVER'
                return False
            else:
                return True






