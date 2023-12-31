'''Modified from # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
'''
import os.path as osp
import tempfile
import os
from mmdet3d.datasets import NuScenesDataset
import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import wandb

# __all__ = ['DetNuscEvaluator']


class DetNuscEvaluator():
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE',
    }

    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(
        self,
        class_names,
        eval_version='detection_cvpr_2019',
        data_root='/usr/src/nuscenes',
        version='v1.0-trainval',
        modality=dict(use_lidar=False,
                      use_camera=True,
                      use_radar=False,
                      use_map=False,
                      use_external=False),
        output_dir=None,
    ) -> None:
        self.eval_version = eval_version
        self.data_root = data_root
        if self.eval_version is not None:
            from nuscenes.eval.detection.config import config_factory

            self.eval_detection_configs = config_factory(self.eval_version)
        self.version = version
        self.class_names = class_names
        self.modality = modality
        self.output_dir = output_dir

    def _evaluate_single(self, 
                         result_path,
                         lm=None,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version,
                        dataroot=self.data_root,
                        verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(nusc,
                                 config=self.eval_detection_configs,
                                 result_path=result_path,
                                 eval_set=eval_set_map[self.version],
                                 output_dir=output_dir,
                                 verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics

        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for class_name in self.class_names:
            for k, v in metrics['label_aps'][class_name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, class_name,
                                                 k)] = val
            for k, v in metrics['label_tp_errors'][class_name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, class_name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        #! WandB Logs
        if lm is not None:
            wandb.log({f'{lm.trainer.state.stage}/metrics//NDS': metrics['nd_score'],\
                 f'{lm.trainer.state.stage}/metrics/mAP': metrics['mean_ap'], 'epoch': lm.current_epoch})
            if lm.current_epoch % 6 ==0:
                mmcv.dump(metrics, osp.join(output_dir, f'metrics_summary_{lm.current_epoch}ep.json'))
        return detail

    def format_results(self, results, img_metas, lidar_cali, result_names=['pts_bbox'], jsonfile_prefix=None,lm=None):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, img_metas, lidar_cali, self.output_dir, lm)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in ['pts_bbox']:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, img_metas, lidar_cali, tmp_file_, lm)})
        return result_files, tmp_dir

    def evaluate(
        self,
        lm,
        results,
        img_metas,
        lidar_cali,
        metric='bbox',
        logger=None,
        jsonfile_prefix=None,
        result_names=['pts_bbox'],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """

        result_files, tmp_dir = self.format_results(results, img_metas,lidar_cali,
                                                    result_names,
                                                    jsonfile_prefix,lm)
        if isinstance(result_files, dict): #! 여기
            metrics = []
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                metrics.append(self._evaluate_single(result_files[name], lm=lm))
            return metrics
        elif isinstance(result_files, str):
            res = self._evaluate_single(result_files, lm=lm)
            return res

        if tmp_dir is not None:
            tmp_dir.cleanup()
        

    def _format_bbox(self, results, img_metas,lidar_cali, jsonfile_prefix=None, lm=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.class_names

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            # print(boxes)
            sample_token = img_metas[sample_id]['token']
            boxes = lidar_nusc_box_to_global(img_metas[sample_id], boxes, lidar_cali[sample_id],
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            # print(boxes)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        if (lm is not None) and (lm.current_epoch % 6 ==0):
            res_path = osp.join(jsonfile_prefix, f'results_nusc_{lm.current_epoch}ep.json') #! Epoch 
        else:
            res_path = osp.join(jsonfile_prefix, 'results_nusc.json') #! Epoch
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        # print('copy results_nusc.json to /data/detr3d/results_nusc.json')
        # os.system("sudo cp " + res_path +" /data/detr3d/")
        return res_path
    
    
def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection[0]
    scores = detection[1].numpy()
    labels = detection[2].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = Box(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(img_metas,
                             boxes,
                             lidar_cali,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.
    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'
    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        # print(lidar_calibrated['rotation'].shape)
        # box.rotate(pyquaternion.Quaternion(lidar_cali['rotation']))
        # box.translate(np.array(lidar_cali['translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        
        box.rotate(pyquaternion.Quaternion(img_metas['ego2global_rotation'].cpu().numpy()))
        box.translate(np.array(img_metas['ego2global_translation'].cpu().numpy()))
        box_list.append(box)
        
    return box_list