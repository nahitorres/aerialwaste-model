# -*- coding: utf-8 -*-

from importlib import import_module

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.image_wrapper import ImageWrapper
from utils.imutils import (get_strided_up_size, pre_process_image,
                          process_image_for_cams, rescale_image)


class ImageProcessor:
    """Class used to execute trained models on single images.

    You can given an input image you can calculate the classification scores
    and the class activation maps (CAMs).

    Parameters
    ----------
    cats : list of str
        List containing the names of the classes object of the classification
        and class activation mapping tasks.
    state_dict_path : str
        Full path to the `.pth` file containing the model weights saved after
        training.
    model : str, optional
        Python module containing the trained model,
        by default "net.resnet50_cam".
    scales : tuple of floats, optional
        Scales used to generate the rescaled images of the original image
        that will be used to compute a more precise global class
        activation map (CAM), by default (1.0, 0.5, 1.5, 2.0).
    gpu : int, optional
        Id of the GPU used to perform the classification and the class
        activation mapping tasks, by default 0.
    """
    """list of str: Names of the intermediate feature pyramid network layers
    (FPN)"""
    FPN_INTERMEDIATE_FEATURE_MAPS_NAMES = ["p5", "p4", "p3", "p2"]

    """str: Class name of the model used for classification."""
    NET_MODEL_CLASS_NAME = "Net"

    """str: Class name of the model used for computing the global
    class activation maps (CAMs) for the image."""
    CAM_MODEL_CLASS_NAME = "CAM"

    """str: Class name of the model used for computing the class activation maps
    (CAMs) for each of the intermediate feature maps obtained from the
    feature pyramid network (FPN)."""
    CAM_SCALES_MODEL_CLASS_NAME = "CAM_SCALES"

    """str: Class name of the model used for computing the classification
    and the global class activation maps (CAMs) for the image."""
    CAM_PRED_MODEL_CLASS_NAME = "CAM_PRED"

    def __init__(self, cats, state_dict_path, model="net.resnet50_cam",
                 scales=(1.0, 0.5, 1.5, 2.0), gpu=0):
        self.__cats = cats
        self.__num_cats = len(cats)
        self.__state_dict_path = state_dict_path
        self.__clear_models()
        self.__model = model
        self.__scales = scales
        torch.cuda.set_device(gpu)

    @property
    def cats(self):
        """list of str: Names of the target categories."""
        return self.__cats

    @cats.setter
    def cats(self, cats):
        self.__cats = cats
        self.__num_cats = len(cats)

    @property
    def model(self):
        """str: Python path of the module containing
        the neural network model."""
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model
        self.__clear_models()

    @property
    def num_cats(self):
        """int: Number of categories."""
        return self.__num_cats

    @property
    def scales(self):
        """tuple of floats: Rates at which the input image is rescaled."""
        return self.__scales

    @scales.setter
    def scales(self, scales):
        self.__scales = scales

    @property
    def state_dict_path(self):
        """str: Full path where the `.pth` file with the saved weights and biases
        of the neural network can be found.

        When a new path is set, if previous models were already loaded, they
        are cleared out."""
        return self.__state_dict_path

    @state_dict_path.setter
    def state_dict_path(self, state_dict_path):
        self.__state_dict_path = state_dict_path
        self.__clear_models()

    def current_gpu(self):
        """Returns the GPU on which the computations are performed.

        Returns
        -------
        int
            ID of the GPU in use.
        """
        return torch.cuda.current_device()

    def execute_cams(self, image):
        """Runs the Neural Network with the loaded weights and biases on the
        target image to get the Class Activation Maps (CAMs) for each category.

        Parameters
        ----------
        image : str or numpy.ndarray
            The target image on which the computations will be executed. It
            can be both the path were the image file is placed or the
            array-like representation of it.

        Returns
        -------
        ImageWrapper
            The image wrapper containing the computed image CAMs.
        """
        image = self.__load_image(image)
        image_wrapper = ImageWrapper(image, self.__cats)
        # Lazy-loading of the model.
        if self.__cam_model is None:
            self.__cam_model =\
                self.__load_model(self.CAM_MODEL_CLASS_NAME)

        image_size = [
            torch.tensor([image_wrapper.height]),
            torch.tensor([image_wrapper.width]),
        ]

        scaled_images = self.__compute_scaled_images_for_cams(image)
        scaled_tensors = [torch.from_numpy(si) for si in scaled_images]

        with torch.no_grad():
            self.__cam_model.cuda()
            image_labels = torch.from_numpy(np.ones(self.num_cats))
            valid_cat = torch.nonzero(image_labels, as_tuple=True)[0]
            strided_up_size = get_strided_up_size(image_size, 16)

            # The outputs list contains as many tensors as the number of
            # scales and each of them will be a 3-d tensor.

            # For example if the scales are (1.0, 0.5, 1.5, 2.0) the shapes of
            # the output tensors will be:
            # [
            #    (1, image_size*1.0, image_size*1.0),
            #    (1, image_size*0.5, image_size*0.5),
            #    (1, image_size*1.5, image_size*1.5),
            #    (1, image_size*2.0, image_size*2.0)
            # ]
            outputs = [self.__cam_model(t.cuda(non_blocking=True))
                       for t in scaled_tensors]

            # I add a dimension to each of the output tensors. For example, if
            # the previous size of the output tensor was:
            # (1, image_size*1.0, image_size*1.0)
            # It becomes:
            # (1, 1, image_size*1.0, image_size*1.0)
            # This is necessary for the next upsampling operation.
            unsqueezed_outputs = [o.unsqueeze(dim=0) for o in outputs]

            # Upsampling each of the scaled output tensors to the target
            # `strided_up_size`. After this operation I have a number of
            # output tensors all with size:
            # (1, 1, strided_up_size, strided_up_size)
            interpolated_outputs = [F.interpolate(
                uo, strided_up_size, mode="bilinear", align_corners=False)
                for uo in unsqueezed_outputs]

            # Concatenating my interpolated output tensors along the first
            # dimension. If I had 4 interpolated output tensors, each with
            # size: (1, 1, strided_up_size, strided_up_size), I end up with a
            # single tensor with size: (4, 1, strided_up_size, strided_up_size)

            # Note: concatenating joins a sequence of tensors along an
            # existing axis while stacking joins a sequence of tensors along a
            # new axis.
            concat_outputs = torch.cat(interpolated_outputs)

            # Summing together all these tensors.
            summed_outputs = torch.sum(concat_outputs, dim=0)

            # Boolean masking to get the valid class activation maps.
            cams = summed_outputs[valid_cat]

            # Normalizing.
            cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5

        image_wrapper.global_cams = cams.cpu().data.numpy()

        return image_wrapper

    def execute_cams_pred(self, image):
        """Runs the Neural Network with the loaded weights and biases on the
        target image to get both the Class Activation Maps (CAMs) and the
        classification scores for each category.

        Parameters
        ----------
        image : str or numpy.ndarray
            The target image on which the computations will be executed. It
            can be both the path were the image file is placed or the
            array-like representation of it.

        Returns
        -------
        ImageWrapper
            The image wrapper containing the classification results and the
            computed image CAMs.
        """
        image = self.__load_image(image)
        image_wrapper = ImageWrapper(image, self.__cats)
        # Lazy-loading of the model.
        if self.__cam_pred_model is None:
            self.__cam_pred_model =\
                self.__load_model(self.CAM_PRED_MODEL_CLASS_NAME)

        image_size = [
            torch.tensor([image_wrapper.height]),
            torch.tensor([image_wrapper.width]),
        ]

        scaled_images = self.__compute_scaled_images_for_cams(image)
        scaled_tensors = [torch.from_numpy(si) for si in scaled_images]

        with torch.no_grad():
            self.__cam_pred_model.cuda()
            image_labels = torch.from_numpy(np.ones(self.num_cats))
            valid_cat = torch.nonzero(image_labels, as_tuple=True)[0]
            strided_up_size = get_strided_up_size(image_size, 16)
            results = [self.__cam_pred_model(t.cuda(non_blocking=True))
                       for t in scaled_tensors]
            outputs = [a for a, b in results]
            scores = [torch.sigmoid(b).cpu().numpy() for a, b in results]
            cams = [F.interpolate(torch.unsqueeze(o, 1), strided_up_size,
                                  mode="bilinear", align_corners=False)
                    for o in outputs]
            cams = torch.sum(torch.stack(cams, 0), 0)[:, 0,
                                                      :image_size[0],
                                                      :image_size[1]]
            cams = cams[valid_cat]
            cams /= F.adaptive_max_pool2d(cams, (1, 1)) + 1e-5

        image_wrapper.global_cams = cams.cpu().data.numpy()
        image_wrapper.classification_scores = scores[0][0]
        return image_wrapper

    def get_model(self):
        if self.__classification_model is None:
            self.__classification_model =\
                self.__load_model(self.NET_MODEL_CLASS_NAME)
        return self.__classification_model
    
    def has_intermediate_cams(self):
        try:
            if self.__cam_scales_model is None:
                self.__cam_scales_model =\
                    self.__load_model(self.CAM_SCALES_MODEL_CLASS_NAME)
                return True
        except Exception as e:
            return False

    def execute_cams_scales(self, image):
        """Runs the Neural Network with the loaded weights and biases on the
        target image to get the Class Activation Maps (CAMs) for each of the
        intermediate scales produced by the Feature Pyramid Network (FPN).

        Parameters
        ----------
        image : str or numpy.ndarray
            The target image on which the computations will be executed. It
            can be both the path were the image file is placed or the
            array-like representation of it.

        Returns
        -------
        ImageWrapper
            The image wrapper containing the computed
            image CAMs for all the scales.
        """
        image = self.__load_image(image)
        image_wrapper = ImageWrapper(image, self.__cats)
        # Lazy-loading of the model.
        if self.__cam_scales_model is None:
            self.__cam_scales_model =\
                self.__load_model(self.CAM_SCALES_MODEL_CLASS_NAME)

        image_size = [
            torch.tensor([image_wrapper.height]),
            torch.tensor([image_wrapper.width]),
        ]
        image = pre_process_image(image)
        image = process_image_for_cams(image)
        image_tensor = torch.from_numpy(image)

        with torch.no_grad():
            self.__cam_scales_model.cuda()
            image_labels = torch.from_numpy(np.ones(self.num_cats))
            valid_cat = torch.nonzero(image_labels, as_tuple=True)[0]
            strided_up_size = get_strided_up_size(image_size, 16)

            # The outputs list contains as many tensors as the number of
            # scales and each of them will be a 3-d tensor.

            # For example if the scales are (1.0, 0.5, 1.5, 2.0) the shapes of
            # the output tensors will be:
            # [
            #    (1, image_size*1.0, image_size*1.0),
            #    (1, image_size*0.5, image_size*0.5),
            #    (1, image_size*1.5, image_size*1.5),
            #    (1, image_size*2.0, image_size*2.0)
            # ]
            outputs = self.__cam_scales_model(
                image_tensor.cuda(non_blocking=True))

            # I add a dimension to each of the output tensors. For example, if
            # the previous size of the output tensor was:
            # (1, image_size*1.0, image_size*1.0)
            # It becomes:
            # (1, 1, image_size*1.0, image_size*1.0)
            # This is necessary for the next upsampling operation.
            unsqueezed_outputs = [o.unsqueeze(dim=0) for o in outputs]

            # Upsampling each of the scaled output tensors to the target
            # `strided_up_size`. After this operation I have a number of
            # output tensors all with size:
            # (1, 1, strided_up_size, strided_up_size)
            interpolated_outputs = [F.interpolate(
                uo, strided_up_size, mode="bilinear", align_corners=False)
                for uo in unsqueezed_outputs]

            # Remove first dimension.
            squeezed_ouputs = [io.squeeze(dim=0)
                               for io in interpolated_outputs]

            # Boolean masking to get the valid class activation maps.
            intermediate_cams = [io[valid_cat] for io in squeezed_ouputs]

            # Normalizing.
            intermediate_cams = [
                ic / F.adaptive_max_pool2d(ic, (1, 1)) + 1e-5
                for ic in intermediate_cams]

            _dict = dict()
            names = ["p5", "p4", "p3", "p2"]
            for n, ic in zip(names, intermediate_cams):
                _dict[n] = ic.cpu().data.numpy()

        image_wrapper.intermediate_cams = _dict

        return image_wrapper

    def execute_classification(self, image):
        """Runs the Neural Network with the loaded weights and biases on the
        target image to get the classification scores for each category.

        Parameters
        ----------
        image : str or numpy.ndarray
            The target image on which the computations will be executed. It
            can be both the path were the image file is placed or the
            array-like representation of it.

        Returns
        -------
        ImageWrapper
            The image wrapper containing the classification results.
        """
        image = self.__load_image(image)
        image_wrapper = ImageWrapper(image, self.__cats)
        # Lazy-loading of the model.
        if self.__classification_model is None:
            self.__classification_model =\
                self.__load_model(self.NET_MODEL_CLASS_NAME)

        with torch.no_grad():
            self.__classification_model.cuda()
            pre_processed_image = pre_process_image(image)
            processed_image = np.expand_dims(pre_processed_image, axis=0)
            processed_tensor = torch.from_numpy(processed_image)
            output = self.__classification_model(
                processed_tensor.cuda(non_blocking=True).requires_grad_())
            scores = torch.nn.Softmax(output[0])
            scores = torch.sigmoid(output[0])

        image_wrapper.classification_scores = scores.cpu().numpy()
        return image_wrapper

    def set_gpu(self, gpu_id):
        """Sets the GPU on which the computations will be executed.

        Parameters
        ----------
        gpu_id: int
            ID of the target GPU.
        """
        torch.cuda.set_device(gpu_id)

    def __clear_models(self):
        """Clears out all the models loaded so far."""
        self.__cam_model = None
        self.__cam_pred_model = None
        self.__cam_scales_model = None
        self.__classification_model = None

    def __compute_scaled_images_for_cams(self, image):
        """Generates all rescaled images of the original image starting from
        the input scales passed in to the constructor."""
        scaled_images = list()
        for s in self.scales:
            if s == 1:
                scaled_image = image
            else:
                scaled_image = rescale_image(image, s, order=3)
            pre_processed_scaled_image = pre_process_image(scaled_image)
            scaled_image_for_cams =\
                process_image_for_cams(pre_processed_scaled_image)
            scaled_images.append(scaled_image_for_cams)

        return scaled_images

    def __load_model(self, class_name):
        """Loads the Neural Network model with the weights and biases
        specified in the `.pth` file located at `state_dict_path`.

        Parameters
        ----------
        class_name : str
            Name of the model class.

        Returns
        -------
        net.resnet50_cam.Net or net.resnet50_cam.CAM:
            Class representing the model.
        """

        model_class = getattr(import_module(self.__model), class_name)
        model = model_class(self.num_cats, pretrained=False)
        model.load_state_dict(torch.load(self.state_dict_path), strict=True)

        model.eval()
        return model

    def __load_image(self, image_src):
        """Load the image from the given source.

        Parameters
        ----------
        image_src : str or np.ndarray
            Path were the image file is placed or array-like representation of
            the image.

        Returns
        -------
        np.ndarray
            Array-like representation of the image.
        """
        if type(image_src) is str:
            image = Image.open(image_src)
            if image.mode not in ('L', 'RGB'):
                image = image.convert('RGB')
            image = np.asarray(image)
        elif type(image_src) is np.ndarray:
            image = image_src
        return image
