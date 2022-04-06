from im2scene.eval import (
    calculate_activation_statistics, calculate_frechet_distance)
from im2scene.training import (
    toggle_grad, compute_grad2, compute_bce, update_average)
from torchvision.utils import save_image, make_grid
import os
import torch
from im2scene.training import BaseTrainer
from tqdm import tqdm
import logging
logger_py = logging.getLogger(__name__)
from torchvision import transforms
import torch.nn.functional as F

from im2scene.common import batch_rot_matrix_to_ht, batch_orth_proj_matrix
from decalib.utils.rotation_converter import batch_rodrigues, batch_orth_proj
from face_parsing.model import BiSeNet as FaceParseNet

class Trainer(BaseTrainer):
    ''' Trainer object for HEADNERF.

    Args:
        model (nn.Module): HEADNERF model
        optimizer (optimizer): generator optimizer object
        optimizer_d (optimizer): discriminator optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        multi_gpu (bool): whether to use multiple GPUs for training
        fid_dict (dict): dicionary with GT statistics for FID
        n_eval_iterations (int): number of eval iterations
        overwrite_visualization (bool): whether to overwrite
            the visualization files
    '''

    def __init__(self, model, optimizer, optimizer_d, device=None,
                 vis_dir=None,
                 multi_gpu=False, fid_dict={},
                 n_eval_iterations=10,
                 overwrite_visualization=True, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.device = device
        self.vis_dir = vis_dir
        self.multi_gpu = multi_gpu

        self.overwrite_visualization = overwrite_visualization
        self.fid_dict = fid_dict
        self.n_eval_iterations = n_eval_iterations

        # Load face segmentation network
        self.face_parse_net = FaceParseNet(n_classes=19)
        self.face_parse_net.cuda()
        self.face_parse_net.load_state_dict(
            torch.load(os.path.join('face_parsing', '79999_iter.pth'))
        )
        self.face_parse_net.eval()

        # TODO: reimplement this?
        #self.vis_dict = model.generator.get_vis_dict(16)

        if multi_gpu:
            self.generator = torch.nn.DataParallel(self.model.generator)
            self.discriminator = torch.nn.DataParallel(
                self.model.discriminator)
            if self.model.generator_test is not None:
                self.generator_test = torch.nn.DataParallel(
                    self.model.generator_test)
            else:
                self.generator_test = None
        else:
            self.generator = self.model.generator
            self.discriminator = self.model.discriminator
            self.generator_test = self.model.generator_test

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def get_codes_from_data(self, data):
        ''' Uses DECA to estimate latent codes and camera pose

        Args:
            data (dict): data dictionary

        Returns:

        '''
        batch_size = data['image'].shape[0]

        deca_img = transforms.Resize(224)(data['image'])  # kut DECA
        code_dict = self.generator.deca.encode(deca_img)
        #latents = (code_dict['shape'], code_dict['exp'], code_dict['tex'], code_dict['light'])
        latents = (code_dict['shape'], code_dict['exp'], code_dict['pose'][:, 3:], code_dict['tex'], code_dict['detail'])

        # TODO: check rotation matrices
        # Convert DECA estimated angle-axis and cam position to homogeneous transform matrix
        rotation_matrices = batch_rodrigues(code_dict['pose'][:, 0:3])
        ht_canonical2world = batch_rot_matrix_to_ht(rotation_matrices)
        ht_world2camera = batch_orth_proj_matrix(code_dict['cam'])
        world_mat = torch.matmul(ht_world2camera, ht_canonical2world)

        # Get a general camera perspective projection matrix
        # TODO: add fov of dataset, for datasets where it is available
        camera_mat = self.generator.camera_matrix.repeat(batch_size, 1, 1)

        return latents, camera_mat, world_mat

    def segment_images(self, images):
        # Segment input image

        img_normalized = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(images)  # normalization for face parser
        img_normalized = transforms.Resize((512, 512))(img_normalized)  # Upsize to expected input size for FaceParse. Note that FaceParse works with other sizes, but performance is a lot worse!

        out = self.face_parse_net(img_normalized)[0]
        output_predictions = out.argmax(dim=1)
        output_predictions = transforms.Resize((256, 256))(output_predictions)  # Downsize back to working size. TODO: read config

        #mask = (output_predictions <= 14) & (output_predictions > 0) | (output_predictions == 17) # hair is 17
        mask = output_predictions > 0
        segmented_img = torch.where(mask.unsqueeze(1), images, torch.tensor([0.], device=self.device))

        return segmented_img, mask

    def train_step(self, data, it=None):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
            it (int): training iteration
        '''

        # TODO: implement multi_gpu
        toggle_grad(self.generator, True)
        self.generator.train()
        self.optimizer.zero_grad()

        latents, camera_mat, world_mat = self.get_codes_from_data(data)

        # Generate image from latent code and camera matrices
        x_fake = self.generator(latent_codes=latents,
                           camera_matrices=(camera_mat, world_mat),
                           not_render_background=True, it=it)

        segmented_img, mask = self.segment_images(data['image'])

        # Compute l2 loss with x_fake only for the mask
        gloss = F.mse_loss(x_fake, mask.unsqueeze(1) * segmented_img)

        gloss.backward()
        self.optimizer.step()

        if self.generator_test is not None:
            update_average(self.generator_test, self.generator, beta=0.999)

        loss_g = gloss.item()

        return {
            'generator': loss_g,
        }

    # def eval_step(self):
    #     ''' Performs a validation step.
    #
    #     Args:
    #         data (dict): data dictionary
    #     '''
    #
    #     gen = self.model.generator_test
    #     if gen is None:
    #         gen = self.model.generator
    #     gen.eval()
    #
    #     x_fake = []
    #     n_iter = self.n_eval_iterations
    #
    #     for i in tqdm(range(n_iter)):
    #         with torch.no_grad():
    #             x_fake.append(gen().cpu()[:, :3])
    #     x_fake = torch.cat(x_fake, dim=0)
    #     x_fake.clamp_(0., 1.)
    #     mu, sigma = calculate_activation_statistics(x_fake)
    #     fid_score = calculate_frechet_distance(
    #         mu, sigma, self.fid_dict['m'], self.fid_dict['s'], eps=1e-4)
    #     eval_dict = {
    #         'fid_score': fid_score
    #     }
    #
    #     return eval_dict

    def visualize(self, data, it=0):
        ''' Visualized the data.

        Args:
            it (int): training iteration
        '''

        latents, camera_mat, world_mat = self.get_codes_from_data(data)

        self.model.generator.eval()
        with torch.no_grad():
            x_fake = self.generator(latent_codes=latents,
                                    camera_matrices=(camera_mat, world_mat),
                                    not_render_background=True,
                                    mode='val').cpu()

        batch_size = data['image'].shape[0]
        x_real = data['image'].cpu()
        images = torch.zeros(batch_size*3, data['image'].shape[1], data['image'].shape[2], data['image'].shape[3])

        segmented_images, _ = self.segment_images(data['image'])

        images[::3, ...] = x_real
        images[1::3, ...] = segmented_images
        images[2::3, ...] = x_fake


        if self.overwrite_visualization:
            out_file_name = 'visualization.png'
        else:
            out_file_name = 'visualization_%010d.png' % it

        image_grid = make_grid(images.clamp_(0., 1.), nrow=3)
        save_image(image_grid, os.path.join(self.vis_dir, out_file_name))
        return image_grid

