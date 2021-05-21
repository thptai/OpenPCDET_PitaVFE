import torch

from .vfe_template import VFETemplate


class PitaVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        # print("/n !!! start")
        # voxel_features, voxel_num_points = batch_dict['voxels'], batch_dict['voxel_num_points']
        # points_mean = voxel_features[:, :, :].sum(dim=1, keepdim=False)
        # normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxel_features)
        # points_mean = points_mean / normalizer
        # print(voxel_features[0])
        # print(points_mean[0])

        #PitaVFE ver1.0, using elongation
        # voxel_features_test, voxel_num_points_test = batch_dict['voxels'], batch_dict['voxel_num_points']
        # voxel_features_intesity_mask = (voxel_features_test[:, :, 3:4] > 0.0).float()
        # voxel_features_elong = (1.0 - voxel_features_test[:, :, 4:5] / 1.5) * voxel_features_intesity_mask
        # voxel_features_elong_sum = voxel_features_elong.sum(dim=1, keepdim=True)
        # voxel_features_elong_ratio = voxel_features_elong / voxel_features_elong_sum
        # voxel_features_intensity = (voxel_features_test[:, :, :] * voxel_features_elong_ratio).sum(dim=1, keepdim=False)
        # points_mean_test = voxel_features_test[:, :, :].sum(dim=1, keepdim=False)
        # normalizer = torch.clamp_min(voxel_num_points_test.view(-1, 1), min=1.0).type_as(voxel_features_test)
        # points_mean_test = points_mean_test / normalizer
        # points_mean_test[:, 3:4]  = voxel_features_intensity[:, 3:4]

        #PitaVFE ver2.0, using coordination
        voxel_features_test, voxel_num_points_test = batch_dict['voxels'], batch_dict['voxel_num_points']
        voxel_features_intesity_mask = (voxel_features_test[:, :, 3:4] > 0.0).float()
        voxel_features_x_axis = voxel_features_test[:, :, 0:1] * voxel_features_intesity_mask
        voxel_features_x_axis = voxel_features_x_axis * voxel_features_x_axis
        voxel_features_y_axis = voxel_features_test[:, :, 1:2] * voxel_features_intesity_mask
        voxel_features_y_axis = voxel_features_y_axis * voxel_features_y_axis
        voxel_features_z_axis = voxel_features_test[:, :, 2:3] * voxel_features_intesity_mask
        voxel_features_z_axis = voxel_features_z_axis * voxel_features_z_axis
        voxel_features_dist = torch.add(voxel_features_x_axis, voxel_features_y_axis)
        voxel_features_dist = torch.add(voxel_features_dist, voxel_features_z_axis, alpha=100)

        voxel_features_dist_sum = voxel_features_dist.sum(dim=1, keepdim=True)
        voxel_features_dist_ratio = voxel_features_dist / voxel_features_dist_sum
        voxel_features_new = (voxel_features_test[:, :, :] * voxel_features_dist_ratio).sum(dim=1, keepdim=False)
        # points_mean_test = voxel_features_new
        # print(voxel_features_test[0])
        # print(voxel_features_x_axis[0])
        # print(voxel_features_y_axis[0])
        # print(voxel_features_z_axis[0])
        # print(voxel_features_dist[0])
        # print(voxel_features_dist_sum[0])
        # print(voxel_features_dist_ratio[0])
        # print(voxel_features_new[0])

        points_mean_test = voxel_features_test[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(voxel_num_points_test.view(-1, 1), min=1.0).type_as(voxel_features_test)
        points_mean_test = points_mean_test / normalizer
        points_mean_test[:, 3:4]  = voxel_features_new[:, 3:4]
        points_mean_test[:, 4:5]  = voxel_features_new[:, 4:5]
        #print(points_mean_test[0])
        # print(points_mean_test[0])
        #print("/n !!! end !!!")
        batch_dict['voxel_features'] = points_mean_test.contiguous()
        return batch_dict
