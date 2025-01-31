import logging
import os
from typing import Annotated, Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode



import inspect
import itertools
import multiprocessing
import os
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional
import importlib
import pkgutil
import SimpleITK as sitk

import numpy as np
try:
  import cv2
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'opencv' Python package. Click OK to install it now."):
    slicer.util.pip_install("opencv-python")
    import cv2
try:
  import torch
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'torch' Python package. Click OK to install it now."):
    slicer.util.pip_install("torch==2.4.0")
    import torch
# torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
try:
  from acvl_utils.cropping_and_padding.padding import pad_nd_image
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'acvl_utils' Python package. Click OK to install it now."):
    slicer.util.pip_install("acvl-utils==0.2")
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
try:
  import batchgenerators
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'batchgenerators' Python package. Click OK to install it now."):
    slicer.util.pip_install("batchgenerators>=0.25")
    import batchgenerators
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, isdir, subdirs, \
    save_json
try:
  import batchgeneratorsv2
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'batchgeneratorsv2' Python package. Click OK to install it now."):
    slicer.util.pip_install("batchgeneratorsv2>=0.2")
    import batchgeneratorsv2
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel
try:
  from tqdm import tqdm
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'tqdm' Python package. Click OK to install it now."):
    slicer.util.pip_install("tqdm")
    from tqdm import tqdm
try:
  from skimage.transform import rescale
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'skimage' Python package. Click OK to install it now."):
    slicer.util.pip_install("scikit-image")
    from skimage.transform import rescale

import sys
import os
current_path = os.path.abspath(__file__)
resources_path = os.path.dirname(current_path)
try:
  import nnunetv2
except ModuleNotFoundError:
  if slicer.util.confirmOkCancelDisplay("This module requires 'nnunetv2' Python package. Click OK to install it now."):
    slicer.util.pip_install("nnunetv2")
    import nnunetv2
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy, preprocessing_iterator_fromfiles, \
    preprocessing_iterator_fromnpy
from nnunetv2.inference.export_prediction import export_prediction_from_logits, \
    convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, \
    compute_steps_for_sliding_window
from nnunetv2.utilities.file_path_utilities import get_output_folder, check_workers_alive_and_busy
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder

class nnUNetPredictor(object):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True):
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        else:
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device

    def initialize_from_trained_model_folder(self, mode, checkpoint_name):
        """
        This is used when making predictions with a trained model
        """
        dataset_json = load_json(join(resources_path, "Resources", "Configs", mode, 'dataset.json'))
        plans = load_json(join(resources_path, "Resources", "Configs", mode, 'plans.json'))
        plans_manager = PlansManager(plans)

        checkpoint = torch.load(join(resources_path, "Resources", "Weights", checkpoint_name),
                                map_location=torch.device('cpu'))
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
            'inference_allowed_mirroring_axes' in checkpoint.keys() else None
        parameters = [checkpoint['network_weights']]

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = 1
        trainer_class = None
        for importer, modname, ispkg in pkgutil.iter_modules([join(resources_path, "Resources", "Configs", "nnUNetTrainer")]):
            if not ispkg:
                m = importlib.import_module('nnunetv2.training.nnUNetTrainer.' + modname)
                if hasattr(m, trainer_name):
                    trainer_class = getattr(m, trainer_name)
                    break
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer. '
                               f'Please place it there (in any .py file)!')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network

        # initialize network with first set of parameters, also see https://github.com/MIC-DKFZ/nnUNet/issues/2520
        network.load_state_dict(parameters[0])

        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if ('nnUNet_compile' in os.environ.keys()) and (os.environ['nnUNet_compile'].lower() in ('true', '1', 't')) \
                and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager, parameters: Optional[List[dict]],
                              dataset_json: dict, trainer_name: str,
                              inference_allowed_mirroring_axes: Optional[Tuple[int, ...]]):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        allow_compile = True
        allow_compile = allow_compile and ('nnUNet_compile' in os.environ.keys()) and (
                    os.environ['nnUNet_compile'].lower() in ('true', '1', 't'))
        allow_compile = allow_compile and not isinstance(self.network, OptimizedModule)
        if isinstance(self.network, DistributedDataParallel):
            allow_compile = allow_compile and isinstance(self.network.module, OptimizedModule)
        if allow_compile:
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    @staticmethod
    def auto_detect_available_folds(model_training_output_dir, checkpoint_name):
        print('use_folds is None, attempting to auto detect available folds')
        fold_folders = subdirs(model_training_output_dir, prefix='fold_', join=False)
        fold_folders = [i for i in fold_folders if i != 'fold_all']
        fold_folders = [i for i in fold_folders if isfile(join(model_training_output_dir, i, checkpoint_name))]
        use_folds = [int(i.split('_')[-1]) for i in fold_folders]
        print(f'found the following folds: {use_folds}')
        return use_folds

    def _manage_input_and_output_lists(self, list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                       output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
                                       folder_with_segs_from_prev_stage: str = None,
                                       overwrite: bool = True,
                                       part_id: int = 0,
                                       num_parts: int = 1,
                                       save_probabilities: bool = False):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(list_of_lists_or_source_folder,
                                                                                       self.dataset_json['file_ending'])
        print(f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[part_id::num_parts]
        caseids = [os.path.basename(i[0])[:-(len(self.dataset_json['file_ending']) + 5)] for i in
                   list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [join(output_folder_or_list_of_truncated_output_files, i) for i in caseids]
        else:
            output_filename_truncated = output_folder_or_list_of_truncated_output_files

        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage, i + self.dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None for i in caseids]
        # remove already predicted files form the lists
        if not overwrite and output_filename_truncated is not None:
            tmp = [isfile(i + self.dataset_json['file_ending']) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + '.npz') for i in output_filename_truncated]
                tmp = [i and j for i, j in zip(tmp, tmp2)]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i] for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in not_existing_indices]
            print(f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                  f'That\'s {len(not_existing_indices)} cases.')
        return list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files

    def predict_from_files(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           num_processes_preprocessing: int = default_num_processes,
                           num_processes_segmentation_export: int = default_num_processes,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, part_id, num_parts,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_lists_or_source_folder,
                                                                                 seg_from_prev_stage_files,
                                                                                 output_filename_truncated,
                                                                                 num_processes_preprocessing)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            seg_from_prev_stage_files: Union[List[str], None],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, seg_from_prev_stage_files,
                                                output_filenames_truncated, self.plans_manager, self.dataset_json,
                                                self.configuration_manager, num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)
        # preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose_preprocessing)
        # # hijack batchgenerators, yo
        # # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # # way we don't have to reinvent the wheel here.
        # num_processes = max(1, min(num_processes, len(input_list_of_lists)))
        # ppa = PreprocessAdapter(input_list_of_lists, seg_from_prev_stage_files, preprocessor,
        #                         output_filenames_truncated, self.plans_manager, self.dataset_json,
        #                         self.configuration_manager, num_processes)
        # if num_processes == 0:
        #     mta = SingleThreadedAugmenter(ppa, None)
        # else:
        #     mta = MultiThreadedAugmenter(ppa, None, num_processes, 1, None, pin_memory=pin_memory)
        # return mta

    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                        np.ndarray,
                                                                                                        List[
                                                                                                            np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):

        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images

        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]

        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]

        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            properties_or_list_of_properties,
            truncated_ofname,
            self.plans_manager,
            self.dataset_json,
            self.configuration_manager,
            num_processes,
            self.device.type == 'cuda',
            self.verbose_preprocessing
        )

        return pp

    def predict_from_list_of_npy_arrays(self,
                                        image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                        segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,
                                                                                                    np.ndarray,
                                                                                                    List[
                                                                                                        np.ndarray]],
                                        properties_or_list_of_properties: Union[dict, List[dict]],
                                        truncated_ofname: Union[str, List[str], None],
                                        num_processes: int = 3,
                                        save_probabilities: bool = False,
                                        num_processes_segmentation_export: int = default_num_processes):
        iterator = self.get_data_iterator_from_raw_npy_data(image_or_list_of_images,
                                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
                                                            properties_or_list_of_properties,
                                                            truncated_ofname,
                                                            num_processes)
        return self.predict_from_data_iterator(iterator, save_probabilities, num_processes_segmentation_export)

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = default_num_processes):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.configuration_manager, self.plans_manager,
                              self.dataset_json, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, self.plans_manager,
                                 self.configuration_manager, self.label_manager,
                                 properties,
                                 save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 output_file_truncated: str = None,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], [segmentation_previous_stage], [image_properties],
                                       [output_file_truncated],
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        if output_file_truncated is not None:
            export_prediction_from_logits(predicted_logits, dct['data_properties'], self.configuration_manager,
                                          self.plans_manager, self.dataset_json, output_file_truncated,
                                          save_or_return_probabilities)
        else:
            ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                              self.configuration_manager,
                                                                              self.label_manager,
                                                                              dct['data_properties'],
                                                                              return_probabilities=
                                                                              save_or_return_probabilities)
            if save_or_return_probabilities:
                return ret[0], ret[1]
            else:
                return ret
    
    def predict_single_npy_array_return(self, input_image: np.ndarray, image_properties: dict,
                                 save_or_return_probabilities: bool = False):
        """
        WARNING: SLOW. ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON.


        input_image: Make sure to load the image in the way nnU-Net expects! nnU-Net is trained on a certain axis
                     ordering which cannot be disturbed in inference,
                     otherwise you will get bad results. The easiest way to achieve that is to use the same I/O class
                     for loading images as was used during nnU-Net preprocessing! You can find that class in your
                     plans.json file under the key "image_reader_writer". If you decide to freestyle, know that the
                     default axis ordering for medical images is the one from SimpleITK. If you load with nibabel,
                     you need to transpose your axes AND your spacing from [x,y,z] to [z,y,x]!
        image_properties must only have a 'spacing' key!
        """
        ppa = PreprocessAdapterFromNpy([input_image], None, [image_properties], None,
                                       self.plans_manager, self.dataset_json, self.configuration_manager,
                                       num_threads_in_multithreaded=1, verbose=self.verbose)
        if self.verbose:
            print('preprocessing')
        dct = next(ppa)

        if self.verbose:
            print('predicting')
        predicted_logits = self.predict_logits_from_preprocessed_data(dct['data']).cpu()

        if self.verbose:
            print('resampling to original shape')
        
        ret = convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits, self.plans_manager,
                                                                            self.configuration_manager,
                                                                            self.label_manager,
                                                                            dct['data_properties'],
                                                                            return_probabilities=
                                                                            save_or_return_probabilities)
        if save_or_return_probabilities:
            return ret[0], ret[1]
        else:
            return ret

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None

        for params in self.list_of_parameters:

            # messing with state dict names...
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            # why not leave prediction on device if perform_everything_on_device? Because this may cause the
            # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
            # this actually saves computation time
            if prediction is None:
                prediction = self.predict_sliding_window_return_logits(data).to('cpu')
            else:
                prediction += self.predict_sliding_window_return_logits(data).to('cpu')

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.configuration_manager.patch_size) < len(image_size):
            assert len(self.configuration_manager.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.configuration_manager.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.configuration_manager.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.configuration_manager.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.configuration_manager.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.configuration_manager.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                predicted_logits[sl] += prediction
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')
                    print("step_size:", self.tile_step_size)
                    print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.configuration_manager.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                               self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits

    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # # check if we need a prediction from the previous stage
        # if self.configuration_manager.previous_stage_name is not None:
        #     assert folder_with_segs_from_prev_stage is not None, \
        #         f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
        #         f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
        #         f' they are located via folder_with_segs_from_prev_stage'

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                None, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(seg_from_prev_stage_files)

        ret = []
        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()

            if of is not None:
                export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                  self.dataset_json, of, save_probabilities)
            else:
                ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                     self.configuration_manager, self.label_manager,
                     data_properties,
                     save_probabilities))

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret
    
    def my_predict(self,
                    # list_of_lists_or_source_folder: Union[str, List[List[str]]],
                    # output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                    data, data_properties, 
                    save_probabilities,
                    overwrite):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        ret = []
        
        data, seg = preprocessor.run_case_npy(data, None, data_properties, 
                                      self.plans_manager, self.configuration_manager,
                                      self.dataset_json)

        print(f'perform_everything_on_device: {self.perform_everything_on_device}')

        prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()

        ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                self.configuration_manager, self.label_manager,
                data_properties,
                save_probabilities))

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


def window_transform_2d(dcm_2d_array, window_width, window_center, normal=False):
    """
    window_width: 窗位
    window_center: 窗宽
    """
    min_window = float(window_center) - 0.5 * float(window_width)
    new_2d_array = (dcm_2d_array - min_window) / float(window_width)
    new_2d_array[new_2d_array < 0] = 0
    new_2d_array[new_2d_array > 1] = 1
    if not normal:
        new_2d_array = (new_2d_array * 255).astype('uint8')
    return new_2d_array

def window_transform_3d(dcm_3d_array, window_width, window_center, low_slice_num=0, high_slice_num=0, normal=False) :
    
    if(high_slice_num == 0):
        high_slice_num = len(dcm_3d_array)
    
    for slice_num in range(low_slice_num, high_slice_num):
        dcm_3d_array[slice_num] = window_transform_2d(dcm_3d_array[slice_num], window_width, window_center, normal)

    return dcm_3d_array


def harris_corner_detection(img, gamma):
    # print(gamma)
    img = img.astype("uint8")
    img = np.float32(img)
    width, height = img.shape
    
    Harris_detector = cv2.cornerHarris(img, 2, 3, 0.04)

    dst = Harris_detector
    thres = gamma * dst.max()
    # print('thres =', thres)
    gray_img = deepcopy(img)
    gray_img[dst <= thres] = 0
    gray_img[dst > thres] = 255
    gray_img = gray_img.astype("uint8")

    coor = np.array([])
    for i in range(width):
        for j in range(height):
            if gray_img[i][j] == 255: 
                coor = np.append(coor, [i, j], axis=0) 
    coor = np.reshape(coor, (-1, 2))
    return gray_img, coor

def get_cross_section(dicom) :
    dcm_3d_array = window_transform_3d(dicom, window_width=1700, window_center=1500).astype(np.uint8)

    # Maximum Intensity Projection
    mip_img = np.max(dcm_3d_array, axis=2)
    mip_img[mip_img != 255] = 0

    # open and close
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_OPEN, kernel, iterations=1)
    mip_img = cv2.morphologyEx(mip_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # get roi slice base on corner detection
    y_coords = np.where(mip_img == 255)[0]
    y_coords = y_coords[(y_coords > 50) & (y_coords < 300)]
    average_y = np.mean(y_coords)

    bi_img, coordinates = harris_corner_detection(mip_img, gamma=0.25) 
    index = np.argsort(coordinates[:, 1]) 
    coordinates = coordinates[index] 
    del_idx = np.array([], dtype=np.int16)
    for i in range(coordinates.shape[0]):
        x = int(coordinates[i, 0])
        y = int(coordinates[i, 1])
        if np.count_nonzero(mip_img[x-1, (y-1):(y+2)]) >= 2 or x >= 250 or x < 50:
            del_idx = np.append(del_idx, int(i))
    
    coordinates = np.delete(coordinates, del_idx, axis=0) 
    coordinates = coordinates.astype(int)
    ori_coordinates = coordinates
    
    # find anomalies
    coordinates_x = coordinates[:, 1]
    coordinates_y = coordinates[:, 0]
    data_std = np.std(coordinates_y)
    data_mean = np.mean(coordinates_y)
    anomaly = data_std * 3

    coordinates = []
    for num in coordinates_y:
        if num <= data_mean + anomaly and num >= data_mean - anomaly :
            coordinates.append(num)

    cs_upper, cs_lower = np.min(coordinates), np.max(coordinates)
    cs_front, cs_rear = np.min(coordinates_x), np.max(coordinates_x)
    # cs_y = int(np.mean(coordinates))
    cs_y = int(average_y)
    # get roi slice base on corner detection
    x_coords = np.where(mip_img == 255)[1]
    # cs_x = int(np.mean(coordinates_x))
    cs_front = int(np.min(x_coords))
    
    cs_front -= 16
    if cs_front < 0 :
        cs_front = 0
    cs_rear = cs_front + 192
    
    cs_upper = cs_y - 96
    cs_lower = cs_y + 96
    
    dcm_3d_array = dcm_3d_array[cs_upper:cs_lower, cs_front:cs_rear, :]
    
    # Maximum Intensity Projection
    mip_img = np.max(dcm_3d_array, axis=1)
    mip_img[mip_img != 255] = 0
    
    teeth_area = np.max(mip_img, axis=0)
    non_zero_area = np.where(teeth_area != 0)[0]
    teeth_min, teeth_max = non_zero_area.min(), non_zero_area.max()
    
    cs_left = (teeth_min + teeth_max) // 2 - 108
    cs_right = (teeth_min + teeth_max) // 2 + 108
        
    return cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right


#
# ImplantGenerator
#


class ImplantGenerator(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("ImplantGenerator")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
            This is an example of scripted loadable module bundled in an extension.
            See more information in <a href="https://github.com/organization/projectname#Generator">module documentation</a>.
            """)
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
            This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
            and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
            """)

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)
        
        print(
        "\n#######################################################################\nPlease cite the following papers "
        "when using Implant Generator:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n "
        "Beichen, Wen et al. (2024). "
        "Intelligent virtual dental implant placement via 3D segmentation strategy"
        "xx, xx\n#######################################################################\n")


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # ImplantGenerator1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ImplantGenerator",
        sampleName="ImplantGenerator1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "ImplantGenerator1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="ImplantGenerator1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="ImplantGenerator1",
    )

    # ImplantGenerator2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="ImplantGenerator",
        sampleName="ImplantGenerator2",
        thumbnailFileName=os.path.join(iconsPath, "ImplantGenerator2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="ImplantGenerator2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="ImplantGenerator2",
    )


#
# ImplantGeneratorParameterNode
#


@parameterNodeWrapper
class ImplantGeneratorParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


#
# ImplantGeneratorWidget
#


class ImplantGeneratorWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/ImplantGenerator.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = ImplantGeneratorLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[ImplantGeneratorParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# ImplantGeneratorLogic
#


class ImplantGeneratorLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return ImplantGeneratorParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        if torch.cuda.is_available() :
            device = "cuda"
        else :
            device = "cpu"
        spacing = 1.0

        if device == 'cpu':
            # let's allow torch to use hella threads
            import multiprocessing
            torch.set_num_threads(multiprocessing.cpu_count())
            device = torch.device('cpu')
        elif device == 'cuda':
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')
        # print(device)
        
        # implant locator process
        #########################
        predictor_loc = nnUNetPredictor(tile_step_size=0.5,
                                    use_gaussian=True,
                                    use_mirroring=True,
                                    perform_everything_on_device=True,
                                    device=device,
                                    verbose=False,
                                    verbose_preprocessing=False,
                                    allow_tqdm=False)
        predictor_loc.initialize_from_trained_model_folder("Locator", 'checkpoint_locator.pth')
        # predictor.predict_from_files_sequential(input, output, False, overwrite=True)
        
        images = []
        spacings = []
        origins = []
        directions = []
        spacings_for_nnunet = []
        
        itk_array = slicer.util.arrayFromVolume(inputVolume)
        dicom = deepcopy(itk_array)
        cs_upper, cs_lower, cs_front, cs_rear, cs_left, cs_right = get_cross_section(dicom)
        dicom = deepcopy(itk_array)
        dicom = window_transform_3d(dicom, window_width=4000, window_center=1000).astype(np.uint8)
        dicom_part = dicom[cs_upper:cs_lower, cs_front:cs_rear, cs_left:cs_right]
        dicom_part = rescale(dicom_part, 0.5, order=1, preserve_range=True)
        
        itk_image = sitk.GetImageFromArray(dicom_part)
        spacings.append(itk_image.GetSpacing())
        origins.append(itk_image.GetOrigin())
        directions.append(itk_image.GetDirection())
        npy_image = sitk.GetArrayFromImage(itk_image)
        if npy_image.ndim == 2:
            # 2d
            npy_image = npy_image[None, None]
            max_spacing = max(spacings[-1])
            spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
        elif npy_image.ndim == 3:
            # 3d, as in original nnunet
            npy_image = npy_image[None]
            spacings_for_nnunet.append(list(spacings[-1])[::-1])
        elif npy_image.ndim == 4:
            # 4d, multiple modalities in one file
            spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
            pass
        else:
            raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

        images.append(npy_image)
        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        data, data_properties = np.vstack(images, dtype=np.float32, casting='unsafe'), dict
        
        predict = predictor_loc.my_predict(data, data_properties, False, True)[0]
        
        # get central position of cylinder
        cylinder = np.array(np.where(predict == 1))
        midx = np.min(cylinder[0]) + np.max(cylinder[0]) + cs_upper
        midy = np.min(cylinder[1]) + np.max(cylinder[1]) + cs_front
        midz = np.min(cylinder[2]) + np.max(cylinder[2]) + cs_left
        dicom = deepcopy(itk_array)
        dicom = window_transform_3d(dicom, window_width=4000, window_center=1000).astype(np.uint8)
        dicom = dicom[midx-48:midx+48, midy-48:midy+48, midz-48:midz+48]

        # Create a new ROI 
        def convertIJKToRAS(volumeNode, ijk_coords):
            ijkToRasMatrix = vtk.vtkMatrix4x4()
            volumeNode.GetIJKToRASMatrix(ijkToRasMatrix)
            ijk_coords_hom = np.array([*ijk_coords, 1])  # 扩展为齐次坐标 (i, j, k, 1)
            ras_coords_hom = np.dot(np.array([ijkToRasMatrix.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4), ijk_coords_hom)
            return tuple(ras_coords_hom[:3])  # 取前三个值作为 RAS 坐标

        roiNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "Missing Region")
        roiNode.SetCenter(convertIJKToRAS(inputVolume, (midz, midy, midx)))
        roiNode.SetSize((96*0.3, 96*0.3, 96*0.3))
        roiNode.SetDisplayVisibility(True)
        slicer.modules.markups.logic().JumpSlicesToNthPointInMarkup(roiNode.GetID(), 0, True)
        
        
        # implant generator process
        ###########################

        predictor_gen = nnUNetPredictor(tile_step_size=0.5,
                                    use_gaussian=True,
                                    use_mirroring=True,
                                    perform_everything_on_device=True,
                                    device=device,
                                    verbose=False,
                                    verbose_preprocessing=False,
                                    allow_tqdm=False)
        predictor_gen.initialize_from_trained_model_folder("Generator", 'checkpoint_generator.pth')
        # predictor.predict_from_files_sequential(input, output, False, overwrite=True)
        
        images = []
        spacings = []
        origins = []
        directions = []
        spacings_for_nnunet = []
        
        # itk_image = sitk.ReadImage(input_image
        # itk_array = slicer.util.arrayFromVolume(inputVolume)
        itk_image = sitk.GetImageFromArray(dicom)
        spacings.append(itk_image.GetSpacing())
        origins.append(itk_image.GetOrigin())
        directions.append(itk_image.GetDirection())
        npy_image = sitk.GetArrayFromImage(itk_image)
        if npy_image.ndim == 2:
            # 2d
            npy_image = npy_image[None, None]
            max_spacing = max(spacings[-1])
            spacings_for_nnunet.append((max_spacing * 999, *list(spacings[-1])[::-1]))
        elif npy_image.ndim == 3:
            # 3d, as in original nnunet
            npy_image = npy_image[None]
            spacings_for_nnunet.append(list(spacings[-1])[::-1])
        elif npy_image.ndim == 4:
            # 4d, multiple modalities in one file
            spacings_for_nnunet.append(list(spacings[-1])[::-1][1:])
            pass
        else:
            raise RuntimeError(f"Unexpected number of dimensions: {npy_image.ndim} in file {f}")

        images.append(npy_image)
        spacings_for_nnunet[-1] = list(np.abs(spacings_for_nnunet[-1]))

        dict = {
            'sitk_stuff': {
                # this saves the sitk geometry information. This part is NOT used by nnU-Net!
                'spacing': spacings[0],
                'origin': origins[0],
                'direction': directions[0]
            },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': spacings_for_nnunet[0]
        }
        data, data_properties = np.vstack(images, dtype=np.float32, casting='unsafe'), dict
        
        predict = predictor_gen.my_predict(data, data_properties, False, True)[0]
        
        outputVolume = slicer.modules.volumes.logic().CloneVolume(slicer.mrmlScene, inputVolume, "outputVolume")
        # slicer.util.setSliceViewerLayers(background=outputVolume)

        
        # spacing = [1.0, 1.0, 1.0]
        # origin = [0.0, 0.0, 0.0]   # 设置体积起点位置 (R, A, S)
        # dimensions = predict.shape[::-1]  # 转换为 (X, Y, Z) 顺序
        # scalar_type = vtk.VTK_INT

        # image_data = slicer.vtkOrientedImageData()
        # image_data.SetDimensions(dimensions)
        # image_data.AllocateScalars(scalar_type, 1)
        # outputVolume.SetAndObserveImageData(image_data)
        # outputVolume.SetSpacing(spacing)
        # outputVolume.SetOrigin(origin)
        
        # outputVolume.SetSpacing(inputVolume.GetSpacing())
        # outputVolume.SetOrigin(inputVolume.GetOrigin())
        # # outputVolume.SetIJKToRASMatrix(inputVolume.GetIJKToRASMatrix())
        # image_data = vtk.vtkImageData()
        # image_data.SetDimensions(inputVolume.shape[::-1])
        # image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        # outputVolume.SetAndObserveImageData(image_data)
        outputArray = slicer.util.arrayFromVolume(outputVolume)
        max_dicom = np.max(outputArray)
        predict = np.array(np.where(predict == 1))
        # print(predict.shape)
        predict[0] += (midx - 48)
        predict[1] += (midy - 48)
        predict[2] += (midz - 48)
        predict = predict.T
        # print(predict)
        for p in predict :
            outputArray[p[0]][p[1]][p[2]] = max_dicom
        # np.copyto(outputArray, itk_array)

        # Notify Slicer that the volume has been updated
        slicer.util.arrayFromVolumeModified(outputVolume)
        
        # Show the result in the 3D Slicer scene
        slicer.util.setSliceViewerLayers(background=outputVolume, foreground=inputVolume, foregroundOpacity=0.0)

        slicer.modules.markups.logic().JumpSlicesToNthPointInMarkup(roiNode.GetID(), 0, True)
        
        # cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        # slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# ImplantGeneratorTest
#


class ImplantGeneratorTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_ImplantGenerator1()

    def test_ImplantGenerator1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("ImplantGenerator1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = ImplantGeneratorLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
