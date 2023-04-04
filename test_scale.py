import numpy as np
import torch

def build_target_normalization(nsteps: int) -> torch.Tensor:
    """Normalization coefficients approximated with 3-rd degree polynomials
    to avoid storing them explicitly, and allow changing the length

    :param nsteps: number of steps to generate normalisation for
    :type nsteps: int
    :return: XY scaling for the steps
    :rtype: torch.Tensor
    """

    normalization_polynomials = np.asarray(
        [
            # x scaling
            [3.28e-05, -0.0017684, 1.8088969, 2.211737],
            # y scaling
            [-5.67e-05, 0.0052056, 0.0138343, 0.0588579],  # manually decreased by 5
        ]
    )
    # assuming we predict x, y and yaw
    coefs = np.stack([np.poly1d(p)(np.arange(nsteps)) for p in normalization_polynomials])
    coefs = coefs.astype(np.float32)
    return torch.from_numpy(coefs).T
scale = build_target_normalization(12)
def _get_non_kin_rescale_params(max_num_scenes: int = 10):
        """Determine the action un-normalization parameters for the non-kinematic model
        from the current dataset in the L5Kit environment.

        :param max_num_scenes: maximum number of scenes to consider to determine parameters
        :return: Tuple of the action un-normalization parameters for non-kinematic model
        """
        scene_ids = list(range(self.max_scene_id)) if not self.overfit else [self.overfit_scene_id]
        if len(scene_ids) > max_num_scenes:  # If too many scenes, CPU crashes
            scene_ids = scene_ids[:max_num_scenes]
        sim_dataset = SimulationDataset.from_dataset_indices(self.dataset, scene_ids, self.sim_cfg)
        return calculate_non_kinematic_rescale_params(sim_dataset)
print(scale)

