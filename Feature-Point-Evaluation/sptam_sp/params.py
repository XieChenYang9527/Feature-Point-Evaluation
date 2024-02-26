import cv2



class Params(object):
    def __init__(self):
        
        self.pnp_min_measurements = 1
        self.pnp_max_iterations = 10
        self.init_min_points = 1

        self.local_window_size = 10
        self.ba_max_iterations = 10

        self.min_tracked_points_ratio = 0.5

        self.lc_min_inbetween_frames = 10   # frames
        self.lc_max_inbetween_distance = 3  # meters
        self.lc_embedding_distance = 22.0
        self.lc_inliers_threshold = 15
        self.lc_inliers_ratio = 0.5
        self.lc_distance_threshold = 2      # meters
        self.lc_max_iterations = 20

        self.ground = False

        self.view_camera_size = 1



class ParamsEuroc(Params):
    
    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=15.0, 
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=2500, scaleFactor=1.1, nlevels=15, edgeThreshold=31)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif config == 'sifi-BRIEF':
            self.feature_detector = cv2.SIFT_create(
                nfeatures=1000, nOctaveLayers=6, contrastThreshold=0.04, edgeThreshold=31, sigma=1.0
            )
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=True)
        elif config == 'fast-brief':
            # 创建FAST特征检测器，并设置参数
            self.feature_detector = cv2.FastFeatureDetector_create(
                threshold=5, nonmaxSuppression=False, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
            )
            # 创建BRIEF描述符提取器
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False
            )
        elif config == 'KAZE-BRIEF':
            self.feature_detector = cv2.KAZE.create(
                upright=False,  # 如果为True，不会计算方向
                threshold=0.001,  # 特征响应阈值
                nOctaves=4,  # 金字塔octaves数量
                nOctaveLayers=4,  # 每个octave的层数
                diffusivity=cv2.KAZE_DIFF_PM_G2  # 扩散方法
            )
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        else:
            raise NotImplementedError

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # self.descriptor_matcher = cv2.FlannBasedMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 2
        self.matching_distance = 25

        self.frustum_near = 0.1  # meters
        self.frustum_far = 50.0

        self.lc_max_inbetween_distance = 4   # meters
        self.lc_distance_threshold = 1.5
        self.lc_embedding_distance = 22.0

        self.view_image_width = 400
        self.view_image_height = 250
        self.view_camera_width = 0.1
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -1
        self.view_viewpoint_z = -10
        self.view_viewpoint_f = 2000


    
        

class ParamsKITTI(Params):
    def __init__(self, config='SIFT-BRIEF'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0, 
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'GFTT-BRISK':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0,
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.BRISK_create()
        elif config == 'KAZE-BRIEF':
            self.feature_detector = cv2.KAZE.create(
                upright=False,  # 如果为True，不会计算方向
                threshold=0.001,  # 特征响应阈值
                nOctaves=4,  # 金字塔octaves数量
                nOctaveLayers=4,  # 每个octave的层数
                diffusivity=cv2.KAZE_DIFF_PM_G2  # 扩散方法
            )
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif config == 'ORB-BRIEF':
            self.feature_detector = cv2.ORB_create(
                nfeatures=2000, scaleFactor=1.2, nlevels=1, edgeThreshold=10)
            self.descriptor_extractor = self.feature_detector
        elif config == 'SIFT-BRIEF':
            self.feature_detector = cv2.SIFT_create(
                nfeatures=1000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6
            )
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif config == 'FAST-BRIEF':
            # 创建FAST特征检测器，并设置参数
            self.feature_detector = cv2.FastFeatureDetector_create(
                threshold=31, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
            )
            # 创建BRIEF描述符提取器
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False
            )

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 3
        self.matching_distance = 30

        self.frustum_near = 0.1    # meters
        self.frustum_far = 1000.0

        self.ground = True

        self.lc_max_inbetween_distance = 50
        self.lc_distance_threshold = 15
        self.lc_embedding_distance = 20.0

        self.view_image_width = 400
        self.view_image_height = 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500   # -10
        self.view_viewpoint_z = -100   # -0.1
        self.view_viewpoint_f = 2000

class ParamsTantarair(Params):
    def __init__(self, config='GFTT-BRIEF'):
        super().__init__()

        if config == 'GFTT-BRIEF':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0,
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)

        elif config == 'GFTT-BRISK':
            self.feature_detector = cv2.GFTTDetector_create(
                maxCorners=1000, minDistance=12.0,
                qualityLevel=0.001, useHarrisDetector=False)

            self.descriptor_extractor = cv2.BRISK_create()
        elif config == 'KAZE-BRIEF':
            self.feature_detector = cv2.KAZE.create(
                upright=False,  # 如果为True，不会计算方向
                threshold=0.001,  # 特征响应阈值
                nOctaves=4,  # 金字塔octaves数量
                nOctaveLayers=4,  # 每个octave的层数
                diffusivity=cv2.KAZE_DIFF_PM_G2  # 扩散方法
            )
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif config == 'ORB-ORB':
            self.feature_detector = cv2.ORB_create(
                nfeatures=2000, scaleFactor=1.2, nlevels=3, edgeThreshold=5)
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        elif config == 'sifi-brief':
            self.feature_detector = cv2.SIFT_create(
                nfeatures=2000, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6
            )
            self.descriptor_extractor = cv2.xfeatures2d.FREAK_create(
                orientationNormalized=False, scaleNormalized=False, patternScale=22, nOctaves=4)
        elif config == 'fast-brief':
            # 创建FAST特征检测器，并设置参数
            self.feature_detector = cv2.FastFeatureDetector_create(
                threshold=10, nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
            )
            # 创建BRIEF描述符提取器
            self.descriptor_extractor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False
            )

        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.matching_cell_size = 15   # pixels
        self.matching_neighborhood = 3
        self.matching_distance = 30

        self.frustum_near = 0.1    # meters
        self.frustum_far = 1000.0

        self.ground = True

        self.lc_max_inbetween_distance = 50
        self.lc_distance_threshold = 15
        self.lc_embedding_distance = 20.0

        self.view_image_width = 400
        self.view_image_height = 130
        self.view_camera_width = 0.75
        self.view_viewpoint_x = 0
        self.view_viewpoint_y = -500   # -10
        self.view_viewpoint_z = -100   # -0.1
        self.view_viewpoint_f = 2000