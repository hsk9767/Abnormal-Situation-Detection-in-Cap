import numpy as np

class Find_assault():
    def __init__(self, Extractor):
        self.find_driver_count = 0
        self.eyes_nose_center_avg = 0
        self.eyes_nose_dist_avg = 0
        self.eyes_nose_dist_mul = 1
        self.area_avg = 0
        self.find_driver_count_limit = 500
        self.weird_state_count = 0
        self.cos_dist_threshold = 0.4
        self.feature_avg = np.zeros((751,)) ##출력 feature 만큼
        self.feature_extractor = Extractor

    def get_eyes_nose_dist_center(self, r_eye, l_eye, nose):
        dist = abs(r_eye[0] - nose[0]) + abs(r_eye[1] - nose[1]) + abs(l_eye[0] - nose[0]) + abs(l_eye[1] - nose[1])
        return self.eyes_nose_dist_mul * dist, (np.array(r_eye) + np.array(l_eye) + np.array(nose))/3

    def get_area(self, pose):
        return pose.bbox[2] * pose.bbox[3]

    def get_cosine_distance(self, a, b):
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        return 1. - np.dot(a, b.T)

    def find_driver(self, current_poses, img):
        ##img bbox 중 가장 오른쪽 x 좌표들만을 다 모아 놓는 것.
        tr_x_list = [ (pose.bbox[0] + pose.bbox[2]) for pose in current_poses ]
        tr_x_arr = np.array([x for x in tr_x_list if x > 2*img.shape[0]/3])
        if tr_x_arr.size > 0:
            most_right_index = np.argmax(tr_x_arr)
        else: ##driver 찾는 과정에서 아무런 오른쪽 부분에 아무런 사람이 존재하지 않으면 다시 찾기 시작해야 함.
            self.find_driver_count = 0
            return False ## return 값은 driver_find 플래그에 대한 bool 값

        ##평균과 거리 구하기
        eyes_nose_dist, eyes_nose_center = self.get_eyes_nose_dist_center(current_poses[most_right_index].keypoints[-4],
                                                                          current_poses[most_right_index].keypoints[-3],
                                                                          current_poses[most_right_index].keypoints[0] )
        ##bbox 넓이 구하기
        area = self.get_area(current_poses[most_right_index])

        ##feature 뽑기
        tl_x, tl_y, width, height = current_poses[most_right_index].bbox
        print("img shape : ", img.shape)
        feature = self.feature_extractor(np.expand_dims(img[tl_y : tl_y + height,tl_x : tl_x + width, : ], 0))
        print("Feature Shape : ", feature.shape)

         ##첫 번째라면 지수이동평균을 측정 값을 주고 첫 번째가 아니라면 지수이동평균에 수식 대입
        if self.find_driver_count == 0:
            self.eyes_nose_dist_avg = eyes_nose_dist
            self.eyes_nose_center_avg = eyes_nose_center
            self.feature_avg = feature
            self.area_avg = area
            self.find_driver_count += 1
            return False
        elif self.find_driver_count > 0:
            self.eyes_nose_dist_avg = 0.99 * eyes_nose_dist + (1 - 0.99) * self.eyes_nose_dist_avg
            self.area_avg = 0.99 * area + (1 - 0.99) * self.area_avg
            dist_from_center_avg = np.sum(np.abs(eyes_nose_center - self.eyes_nose_center_avg))
            ##얼굴이 별로 움직이지 않는 때가 30초 이상 지속될 때의 센터점과 feature 를 basic으로 설정.
            if dist_from_center_avg > self.eyes_nose_dist_avg:
                self.find_driver_count = 0
                return False
            elif self.find_driver_count < self.find_driver_count_limit:
                self.find_driver_count += 1
                self.eyes_nose_center_avg = 0.99 * eyes_nose_center + (1 - 0.99) * self.eyes_nose_center_avg
                self.feature_avg = 0.99 * feature + (1 - 0.99) * self.feature_avg
                return False
            else:
                current_poses[most_right_index].id = 'DRIVER'
                return True

    def is_driver(self, current_poses, img):
        ##img bbox 중 가장 오른쪽 x 좌표들만을 다 모아 놓는 것 중에서 img 의 2/3 을 넘어선 x좌표들을 보유한 애들만.
        tr_x_list = [(pose.bbox[0] + pose.bbox[2]) for pose in current_poses]
        tr_x_arr = np.array([x for x in tr_x_list if x > 2 * img.shape[0] / 3])
        if tr_x_arr.size > 0:
            most_right_index = np.argmax(tr_x_arr)
        else:
            self.weird_state_count += 1
            return False

        _, eyes_nose_center = self.get_eyes_nose_dist_center(current_poses[most_right_index].keypoints[-4],
                                                             current_poses[most_right_index].keypoints[-3],
                                                             current_poses[most_right_index].keypoints[0])
        dist_from_center_avg = np.sum(np.abs(eyes_nose_center - self.eyes_nose_center_avg))
        area = self.get_area(current_poses[most_right_index])

        ##우선, center 좌표가 얼마 움직이지 않았으면 운전자가 그대로 있다고 판다
        if dist_from_center_avg < self.eyes_nose_dist_avg * 2 :##check 할 때에는 좀 더 넓은 오차범위를 둠.
            current_poses[most_right_index].id = 'DRIVER'
            self.weird_state_count = 0
            return True
        elif area < 0.5 * self.area_avg:
            self.weird_state_count += 1
            return False
        else:
            tl_x, tl_y, width, height = current_poses[most_right_index].bbox
            feature = self.feature_extractor(img[tl_y: tl_y + height, tl_x: tl_x + width, :])
            cos_dist_from_avg_feature = self.get_cosine_distance(feature, self.feature_avg)
            if cos_dist_from_avg_feature > self.cos_dist_threshold : ##차이가 많이 나는 경우
                self.weird_state_count += 1
                return False
            else:
                current_poses[most_right_index].id = 'DRIVER'
                self.weird_state_count = 0
                return True
