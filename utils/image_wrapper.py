# -*- coding: utf-8 -*-
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ImageWrapper:
    """Message shown when the image has not been classified."""
    IMG_NOT_CLASSIFIED_ERROR_MSG = "Image not classified."

    """Message shown when the Annotations made for the image has not been 
    loaded."""
    ANNOTATIONS_NOT_LOADED_ERROR_MSG = "Annotations for image not loaded"

    """Message shown when the image global Class Activation Maps (CAMs)
    have not been computed."""
    IMG_GLOBAL_CAMS_NOT_COMPUTED_ERROR_MSG = "Image global CAMs not computed."

    """Message shown when the image intermediate Class Activation Maps (CAMs)
    for each intermediate Feature Pyramid Network (FPN) layer have not been
    computed."""
    IMG_INTERMEDIATE_CAMS_NOT_COMPUTED_ERROR_MSG =\
        "Image intermediate CAMs not computed."

    def __init__(self, image, cats, thresholds=None):
        self.__image = image
        self.__height = self.image.shape[0]
        self.__width = self.image.shape[1]
        self.__cats = cats
        self.__num_cats = len(cats)
        if thresholds is None:
            self.__thresholds = [.5]*self.num_cats
        else:
            assert len(thresholds) == self.num_cats,\
                "The list of thresholds must be the same length as that of "\
                f"categories ({self.num_cats})"
            self.__thresholds = thresholds
        self.__annotations = None
        self.__global_cams = None
        self.__intermediate_cams = None
        self.__classification_scores = None
        self.__predicted_categories = None

    @property
    def annotations(self):
        """numpy.ndarray: array denoting which areas of the image have been labelled
        as an relevant object
        """
        if self.__annotations is None:
            print(self.ANNOTATIONS_NOT_LOADED_ERROR_MSG)
        else:
            return self.__annotations
    
    @annotations.setter
    def annotations(self, annotations):
        self.__annotations = annotations

    @property
    def global_cams(self):
        """list of numpy.ndarray: global Class Activation Maps (CAMs) of the image
        for the target categories.
        """
        if self.__global_cams is None:
            print(self.IMG_GLOBAL_CAMS_NOT_COMPUTED_ERROR_MSG)
        else:
            return self.__global_cams

    @global_cams.setter
    def global_cams(self, global_cams):
        self.__global_cams = global_cams

    @property
    def intermediate_cams(self):
        """dict of {str : list of numpy.ndarray}: dictionary having as keys
        the names of the intermediate layers of Feature Pyramid Network (FPN)
        layers and as values the intermediate Class Activation Maps (CAMs) of
        the image for the target categories for each of these layers.
        """
        if self.__intermediate_cams is None:
            print(self.IMG_INTERMEDIATE_CAMS_NOT_COMPUTED_ERROR_MSG)
        else:
            return self.__intermediate_cams

    @intermediate_cams.setter
    def intermediate_cams(self, intermediate_cams):
        self.__intermediate_cams = intermediate_cams

    @property
    def cats(self):
        """list of str: names of the target categories."""
        return self.__cats

    @property
    def classification_scores(self):
        """numpy.ndarray of float: classification scores of the image for the
        target categories.
        """
        if self.__classification_scores is None:
            print(self.IMG_NOT_CLASSIFIED_ERROR_MSG)
        else:
            return self.__classification_scores

    @classification_scores.setter
    def classification_scores(self, classification_scores):
        self.__classification_scores = classification_scores
        self.__compute_predicted_categories()

    @property
    def height(self):
        """int or float: height of the image."""
        return self.__height

    @property
    def image(self):
        """numpy.ndarray of int: array-like representation of the image."""
        return self.__image

    @property
    def num_cats(self):
        """int: number of categories"""
        return self.__num_cats

    @property
    def predicted_categories(self):
        """dict of {str: float}: which has the names of the predicted
        categories as keys and the corresponding classification score as
        values.
        """
        if self.__predicted_categories is None:
            print(self.IMG_NOT_CLASSIFIED_ERROR_MSG)
        else:
            return self.__predicted_categories

    @property
    def thresholds(self):
        """list of float: classification thresholds.

        Each element of the list must be a float in the range 0-1 (1 excluded)
        and is used to establish whether the category with the corresponding
        index in `cats` is predicted to be in the image or not.
        """
        return self.__thresholds

    @thresholds.setter
    def thresholds(self, thresholds):
        assert len(thresholds) == self.num_cats,\
            "The list of thresholds must be the same length as that of "\
            f"categories ({self.num_cats})"

        self.__thresholds = thresholds

        if self.classification_scores is not None:
            self.__compute_predicted_categories()

    @property
    def width(self):
        """int or float: width of the image."""
        return self.__width

    def save_classification_scores_to_file(self,
                                           file_name,
                                           directory=os.getcwd()):
        """Saves the `classification_scores` in a `.npy` file.

        Parameters
        ----------
        file_name : str
            Name of the output file, by default self.file_name
        directory : str, optional
            Path where the output file is placed, by default the current
            working directory.
        """
        output_path = os.path.join(directory, file_name)
        np.save(output_path, self.classification_scores)

    def show_intermediate_cams(self):
        """Shows the computed global Class Activation Maps (CAMs)
        of the image."""
        if self.__intermediate_cams is None:
            print(self.IMG_INTERMEDIATE_CAMS_NOT_COMPUTED_ERROR_MSG)
        else:
            for layer, cams in self.intermediate_cams.items():
                title = f"Intermediate Layer {layer} CAMs"
                self.__show_cams(cams, title=title)

    def show_global_cams_annotations(self, threshold = None):
        """Shows the computed global Class Activation Maps (CAMs)
        and the Annotations that have been loaded for the image."""
        if self.__global_cams is None:
            print(self.IMG_GLOBAL_CAMS_NOT_COMPUTED_ERROR_MSG)
        elif self.__annotations is None:
            print(self.ANNOTATIONS_NOT_LOADED_ERROR_MSG)
        else:
            self.__show_cams_annotations(self.global_cams, self.__annotations, threshold)

    def show_global_cams(self):
        """Shows the computed global Class Activation Maps (CAMs)
        of the image."""
        if self.__global_cams is None:
            print(self.IMG_GLOBAL_CAMS_NOT_COMPUTED_ERROR_MSG)
        else:
            self.__show_cams(self.global_cams)

    def show_image(self, title=None, figsize=(8, 8)):
        """Shows the original image on screen.

        Parameters
        ----------
        title: string, optional
            Title of the image, by default None
        figsize: tuple of numbers, optional
            Size of the imag in the format(height, width),
            by default(10, 10).
        """
        plt.figure(figsize=figsize)
        plt.imshow(self.image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def __compute_predicted_categories(self):
        """Creates a dictionary of the predicted categories with the
        corresponding classification scores. The dictionary is saved in the
        property `predicted_categories`.
        """
        self.__predicted_categories = {
            self.__cats[i]: s
            for i, s in enumerate(self.__classification_scores)
            if s > self.__thresholds[i]
        }

    def __show_cams(self, cams, title=None):
        """Shows the input class activation maps (CAMs)"""
        cam_arr = cams
        q = len(self.cats) + 1
        columns = 3 if q > q else q
        rows = int(q / columns) + (1 if int(q % columns) > 0 else 0) \
            if q > columns else 1
        fig, axs = plt.subplots(rows, columns, figsize=(20, 20))
        for idx, axi in enumerate(axs.flat):
            if idx == 0:
                axi.imshow(self.image)
                axi.axis("off")
                axi.set_title("Original Image", fontsize=14)
            elif idx < q:
                plt_cam = cam_arr[idx - 1]

                if plt_cam.shape != self.image.shape[:-1]:
                    plt_cam =\
                        np.array(Image.fromarray(
                            plt_cam).resize(self.image.shape[:-1]))
                axi.imshow(self.image)
                axi.imshow(plt_cam, alpha=0.6, vmin=0, vmax=1, cmap="jet")
                axi.set_title(
                    f"Cat {idx} - {self.cats[idx-1]} CAM", fontsize=14)
                axi.axis("off")
            else:
                axi.imshow(np.zeros((1, 1)))
                axi.axis("off")
        fig.suptitle(title, fontsize=24)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
        plt.show()

    def __show_cams_annotations(self, cams, annotations, threshold, title=None):
        """Shows the input class activation maps (CAMs)"""
        cam_arr = cams
        q = len(self.cats) + 1
        columns = 3 if q > q else q
        rows = int(q / columns) + (1 if int(q % columns) > 0 else 0) \
            if q > columns else 1
        fig, axs = plt.subplots(rows, columns, figsize=(20, 20))
        for idx, axi in enumerate(axs.flat):
            if idx == 0:
                axi.imshow(self.image)
                axi.imshow(annotations, alpha=0.3, vmin=0, vmax=1, cmap='binary')
                axi.axis("off")
                axi.set_title("Original Image", fontsize=14)
            elif idx < q:
                plt_cam = cam_arr[idx - 1]
                if not threshold is None:
                    super_threshold_indices = plt_cam < threshold
                    plt_cam[super_threshold_indices] = 0
                if plt_cam.shape != self.image.shape[:-1]:
                    plt_cam =\
                        np.array(Image.fromarray(
                            plt_cam).resize(self.image.shape[:-1]))
                axi.imshow(self.image)
                axi.imshow(plt_cam, alpha=0.6, vmin=0, vmax=1, cmap="jet")
                axi.set_title(
                    f"Cat {idx} - {self.cats[idx-1]} CAM", fontsize=14)
                axi.axis("off")
            else:
                axi.imshow(np.zeros((1, 1)))
                axi.axis("off")
        fig.suptitle(title, fontsize=24)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)
        plt.show()
        
        
