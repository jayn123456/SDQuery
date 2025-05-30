import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC

from shapely.geometry import LineString
from geopandas import GeoSeries
@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class PadMultiViewImageSame2Max(object):

    def __init__(self, size_divisor=None, pad_val=0):
        self.size_divisor = size_divisor
        self.pad_val = pad_val

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        max_h = max([img.shape[0] for img in results['img']])
        max_w = max([img.shape[1] for img in results['img']])
        padded_img = [mmcv.impad(img, shape=(max_h, max_w), pad_val=self.pad_val) for img in results['img']]
        if self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in padded_img]
        
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = None
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape', 'crop_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus', 'lidar2global_rotation'
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1

    def __resize_bboxes(self, results, scale_factor):
        """Resize bboxes according to the scale factor.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        if 'gt_bboxes' in results.keys():
            results['gt_bboxes'] *= scale_factor
        return results

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_scale = np.random.choice(self.scales)

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        cam_intrinsic = [scale_factor @ c2i for c2i in results['cam_intrinsic']]
        results['lidar2img'] = lidar2img
        results['cam_intrinsic'] = cam_intrinsic
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        results['scale_factor'] = [rand_scale for _ in results['img']]

        self.__resize_bboxes(results, rand_scale)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class GridMaskMultiViewImage(object):
    def __init__(self, use_h=True, use_w=True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results

        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]

        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)
        l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)

        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s + l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s + l, ww)
                mask[:,s:t] *= 0

        if self.rotate > 1:
            r = np.random.randint(self.rotate)
            mask = mmcv.imrotate(mask, r)

        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        if self.mode == 1:
            mask = 1-mask
        mask = np.expand_dims(mask, axis=-1)

        for img in imgs:
            img *= mask
        results['img'] = imgs

        return results


@PIPELINES.register_module()
class CropFrontViewImageForAv2(object):

    def __init__(self, crop_h=(356, 1906)):
        self.crop_h = crop_h

    def _crop_img(self, results):
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'][0] = results['img'][0][self.crop_h[0]:self.crop_h[1]]
        results['img_shape'] = [img.shape for img in results['img']]
        results['crop_shape'][0] = np.array([0, self.crop_h[0]])

    def _crop_cam_intrinsic(self, results):
        results['cam_intrinsic'][0][1, 2] -= self.crop_h[0]
        results['lidar2img'][0] = results['cam_intrinsic'][0] @ results['lidar2cam'][0]

    def _crop_bbox(self, results):
        if 'gt_bboxes' in results.keys():
            results['gt_bboxes'][:, 1] -= self.crop_h[0]
            results['gt_bboxes'][:, 3] -= self.crop_h[0]

            mask = results['gt_bboxes'][:, 3] > 0
            results['gt_bboxes'] = results['gt_bboxes'][mask]
            results['gt_labels'] = results['gt_labels'][mask]
            if 'gt_lane_lste_adj' in results.keys():
                results['gt_lane_lste_adj'] = results['gt_lane_lste_adj'][:, mask]

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._crop_img(results)
        self._crop_cam_intrinsic(results)
        self._crop_bbox(results)
        return results
    
@PIPELINES.register_module()
class CustomParametrizeSDMapGraph:
    def __init__(self, method, method_para):
        self.method = method
        self.method_para = method_para

        # define some category mappings
        self.category2id = {
            'road': 0,        # from openlanev2
            'cross_walk': 1,  # from openlanev2
            'side_walk': 1,   # from openlanev2
            'pedestrian': 1,  # set to be sidewalk
            'truck_road': 2, 
            'highway': 3, 
            'residential': 4, 
            'service': 5, 
            'bus_way': 6, 
            'other': 7,
        }

    def __call__(self, results):
        sd_map = results['sd_map']
        sd_map_graph, map_meta = getattr(self, self.method)(sd_map, **self.method_para)
        
        # sd_map_graph: num_polylines x max_num_points x 3
        results["map_graph"] = sd_map_graph
        for key, value in map_meta.items():
            results[key] = value
        # results["map_num_poly_pnts"] = max_num_vec_per_polyline
        return results
    
    def fit_bezier_Endpointfixed(self, points, n_control):
        n_points = len(points)
        A = np.zeros((n_points, n_control))
        t = np.arange(n_points) / (n_points - 1)
        for i in range(n_points):
            for j in range(n_control):
                A[i, j] = self.comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
        A_BE = A[1:-1, 1:-1]
        _points = points[1:-1]
        _points = _points - A[1:-1, 0].reshape(-1, 1) @ points[0].reshape(1, -1) - A[1:-1, -1].reshape(-1, 1) @ points[-1].reshape(1, -1)

        conts = np.linalg.lstsq(A_BE, _points, rcond=None)

        control_points = np.zeros((n_control, points.shape[1]))
        control_points[0] = points[0]
        control_points[-1] = points[-1]
        control_points[1:-1] = conts[0]

        return control_points
    
    def bezier_Endpointfixed(self, sd_map, n_control=2):
        sd_map_graph = []
        for category, polylines in sd_map.items():
            for polyline in polylines:
                res = self.fit_bezier_Endpointfixed(polyline, n_control)
                coeffs = res.flatten()  # n_control * 2, 
                sd_map_graph.append(np.concatenate([coeffs, 
                                                    [self.category2id[category]]], 
                                                    axis=0))
        sd_map_graph = np.concatenate(sd_map_graph, axis=0)
        return sd_map_graph, None
    
    def points_all_with_type(self, sd_map):
        sd_map_graph = []
        num_pnt_per_segment = []
        for category, polylines in sd_map.items():
            # polyline: list npoints, 2
            # category: str
            if len(polylines) > 0:
                for segment in polylines:
                    num_pnt_per_segment.append(segment.shape[0])
                    # segment: npoints, 2
                    sd_map_graph.append(np.concatenate([segment.astype(np.float32), 
                                                        np.ones([segment.shape[0], 1], dtype=np.float32) * self.category2id[category]], 
                                                        axis=1))
        if len(num_pnt_per_segment) == 0:
            num_pnt_per_segment = np.empty([0], dtype=np.int64)
        else:
            num_pnt_per_segment = np.array(num_pnt_per_segment)  # num_polylines, 
        max_num_points = num_pnt_per_segment.max() if len(num_pnt_per_segment) > 0 else 0

        # list of segments: num_polylines, num_points x 3
        # pad at the end
        sd_map_graph = [np.pad(map_graph, ((0, max_num_points - map_graph.shape[0]), (0, 0)), 'constant') for map_graph in sd_map_graph]
        # list of segments: num_polylines, max_num_points x 3

        # then stack: num_polylines x max_num_points x 3
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 else np.zeros([0, max_num_points, 3], dtype=np.float32)
        map_meta = dict(map_num_poly_pnts=num_pnt_per_segment)
        return sd_map_graph, map_meta
    
    def points_onehot_type(self, sd_map):
        num_categories = max(self.category2id.values()) + 1

        sd_map_graph = []
        num_pnt_per_segment = []
        # TODO: polylines can potentially belong to 2 categories, need to re-map it back to the same polyline
        for category, polylines in sd_map.items():
            # polyline: list npoints, 2
            # category: str
            if len(polylines) > 0:
                for segment in polylines:
                    # segment: npoints, 2
                    num_pnt_per_segment.append(segment.shape[0])
                    
                    lane_category_onehot = np.zeros([num_categories], dtype=np.float32)
                    lane_category_onehot[self.category2id[category]] = 1.0
                    sd_map_graph.append(np.concatenate([segment.astype(np.float32),
                                                        lane_category_onehot.reshape(1, -1).repeat(segment.shape[0], axis=0)],
                                                        axis=1))
        if len(num_pnt_per_segment) == 0:
            num_pnt_per_segment = np.empty([0], dtype=np.int64)
        else:
            num_pnt_per_segment = np.array(num_pnt_per_segment)  # num_polylines, 
        max_num_points = num_pnt_per_segment.max() if len(num_pnt_per_segment) > 0 else 0

        # list of segments: num_polylines, num_points x (2 + num_categories)
        # pad at the end
        sd_map_graph = [np.pad(map_graph, ((0, max_num_points - map_graph.shape[0]), (0, 0)), 'constant') for map_graph in sd_map_graph]
        # list of segments: num_polylines, max_num_points x (2 + num_categories)

        # then stack: num_polylines x max_num_points x (2 + num_categories)
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 else np.zeros([0, max_num_points, num_categories + 2], dtype=np.float32)
        map_meta = dict(map_num_poly_pnts=num_pnt_per_segment)
        return sd_map_graph, map_meta
    
    @staticmethod
    def interpolate_line(line, n_points):
        # interpolates a shapely line to n_points
        distances = np.linspace(0, line.length, n_points)
        points = [line.interpolate(distance) for distance in distances]
        return np.stack([point.coords.xy for point in points]).squeeze(-1)

    
    def even_points_onehot_type(self, sd_map, n_points=11):
        num_categories = max(self.category2id.values()) + 1

        sd_map_graph = []
        onehot_category = []

        for category, polylines in sd_map.items():
            # polyline: list, npoints * 2
            # category: str
            if len(polylines) > 0:
                lane_category_onehot = np.zeros([num_categories], dtype=np.float32)
                lane_category_onehot[self.category2id[category]] = 1.0
                # onehot_category: num_polylines * num_categories
                onehot_category.append(lane_category_onehot.reshape(1, -1).repeat(len(polylines), axis=0))

                # interpolate the lines
                lines = GeoSeries(map(LineString, polylines))

                np_lines = [CustomParametrizeSDMapGraph.interpolate_line(line, n_points=n_points).astype(np.float32) \
                             for line in list(lines)]
                sd_map_graph.extend(np_lines)

        # then stack: num_polylines x n_points x 2
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 \
            else np.zeros([0, n_points, 2], dtype=np.float32)
        onehot_category = np.concatenate(onehot_category, axis=0) if len(onehot_category) > 0 \
            else np.zeros([0, num_categories], dtype=np.float32)
        map_meta = dict(onehot_category=onehot_category)
        return sd_map_graph, map_meta
    
    def even_points_by_type(self, sd_map, n_points=11, lane_types=[0, 1], include_category=True):
        # num_categories = max(self.category2id.values()) + 1
        num_categories = len(lane_types)
        lane_old_to_new = {old: new for new, old in enumerate(lane_types)}        

        sd_map_graph = []
        onehot_category = []

        for category, polylines in sd_map.items():
            # polyline: list, npoints * 2
            # category: str
            lanetype_num = self.category2id[category]

            if len(polylines) > 0 and (lanetype_num in lane_types):

                if include_category:
                    lane_category_onehot = np.zeros([num_categories], dtype=np.float32)
                    # lane_category_onehot[lanetype_num] = 1.0
                    lane_category_onehot[lane_old_to_new[lanetype_num]] = 1.0
                    # onehot_category: num_polylines * num_categories
                    onehot_category.append(lane_category_onehot.reshape(1, -1).repeat(len(polylines), axis=0))

                # interpolate the lines
                lines = GeoSeries(map(LineString, polylines))

                np_lines = [CustomParametrizeSDMapGraph.interpolate_line(line, n_points=n_points).astype(np.float32) \
                             for line in list(lines)]
                sd_map_graph.extend(np_lines)

        # then stack: num_polylines x n_points x 2
        sd_map_graph = np.stack(sd_map_graph, axis=0) if len(sd_map_graph) > 0 \
            else np.zeros([0, n_points, 2], dtype=np.float32)
        
        if include_category:
            onehot_category = np.concatenate(onehot_category, axis=0) if len(onehot_category) > 0 \
                else np.zeros([0, num_categories], dtype=np.float32)
        else:
            onehot_category = np.zeros([0, 0], dtype=np.float32)
        map_meta = dict(onehot_category=onehot_category)
        return sd_map_graph, map_meta
