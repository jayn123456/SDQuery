#---------------------------------------------------------------------------------------#
# LaneSegNet: Map Learning with Lane Segment Perception for Autonomous Driving          #
# Source code: https://github.com/OpenDriveLab/LaneSegNet                               #
# Copyright (c) OpenDriveLab. All rights reserved.                                      #
#---------------------------------------------------------------------------------------#

import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES
from shapely.geometry import LineString



@PIPELINES.register_module()
class LaneParameterize3D(object):

    def __init__(self, method, method_para):
        method_list = ['fix_pts_interp']
        self.method = method
        if not self.method in method_list:
            raise Exception("Not implemented!")
        self.method_para = method_para

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        lanes = results['gt_lanes_3d']
        para_lanes = getattr(self, self.method)(lanes, **self.method_para)
        results['gt_lanes_3d'] = para_lanes

        return results

    def fix_pts_interp(self, input_data, n_points=11):
        '''Interpolate the 3D lanes to fix points. The input size is (n_pts, 3).
        '''
        lane_list = []
        for lane in input_data:
            if n_points == 11 and lane.shape[0] == 201:
                lane_list.append(lane[::20].flatten())
            else:
                lane = fix_pts_interpolate(lane, n_points).flatten()
                lane_list.append(lane)
        return np.array(lane_list, dtype=np.float32)


@PIPELINES.register_module()
class GenerateLaneMask(object):
    """Generate mask ground truth for segmentation head
    Args:
        results (dict): Result dict from loading pipeline.
    Returns:
        dict: Instance mask gt is added into result dict.
    """
    def __init__(self, points_num, map_size=[-50, -25, 50, 25], bev_h=100, bev_w=200) -> None:
        self.points_num = points_num
        self.map_size = map_size  # [min_x, min_y, max_x, max_y]
        self.bev_h = bev_h
        self.bev_w = bev_w

    def __call__(self,results):
        results = self._generate_lanesegment_instance_mask(results)
        return results

    def _generate_lanesegment_instance_mask(self, results):
        gt_lanes = np.array(results['gt_lanes_3d']).reshape(-1, 3, self.points_num, 3)
        gt_left_lines = gt_lanes[:, 1]
        gt_right_lines = gt_lanes[:, 2]

        origin = np.array([self.bev_w // 2, self.bev_h // 2])
        scale = np.array([self.bev_w / (self.map_size[2] - self.map_size[0]), self.bev_h / (self.map_size[3] - self.map_size[1])])

        inst_masks = []
        for idx, (left_line, right_line) in enumerate(zip(gt_left_lines, gt_right_lines)):

            segment_boundary = np.concatenate((left_line, right_line[::-1], left_line[0:1]), axis=0)
            mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)

            draw_coor = (segment_boundary[:, :2] * scale + origin).astype(np.int32)
            mask = cv2.fillPoly(mask, [draw_coor], 255)
            bitMask = (mask / 255)
            inst_masks.append(bitMask)

        results['gt_instance_masks'] = inst_masks

        return results
