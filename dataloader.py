"""
This file uses to load MRI images and do pre-processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PILImage
import nibabel as nib
import cv2
import torch
from glob import glob
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
import tarfile
import tempfile
import imageio
import shutil

# Check for GPU
def check_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Uncompress tar files
def uncompress_tar(tar_file_path, uncompressed_dir):
    if not os.path.isfile(tar_file_path):
        raise FileNotFoundError(f"Tar file not found: {tar_file_path}")

    if not os.path.exists(uncompressed_dir):
        os.makedirs(uncompressed_dir)

    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=uncompressed_dir)

    print(f"Uncompressed {tar_file_path} to {uncompressed_dir}")

def load_nifti_image(file_path):
    """Load a NIfTI file."""
    img = nib.load(file_path)
    img_data = img.get_fdata()
    return img_data

def print_middle_mri_slice(file_path, sample_path):
    """Print a middle slice of an MRI image"""

    img_data = load_nifti_image(file_path)
    print('Data Shape: ', img_data.shape)
    img_data_mid_slice = img_data[:,:,img_data.shape[2]//2-1]
    cv2.imwrite(sample_path + f"/slice_{img_data.shape[2]//2-1}.jpg",
                cv2.normalize(img_data_mid_slice,None,0,255,
                               cv2.NORM_MINMAX, cv2.CV_8U))
    print('Save an image successfully!!')

def setup_current_dir():
    """Initialize current directory for project"""
    # Get the current working directory
    current_dir = os.getcwd()
    print("Current directory: ", current_dir)

def create_gif_from_slices(img_data, axis=2, duration=0.1, resize_to=None):
    """
    Create a GIF from slices of a 3D MRI image and display it.

    :param img_data: (numpy.ndarray), 3D image data
    :param axis: (int) 0,1,2; axis along which to slice the 3D image
    :param duration: duraion between frames in the GIF
    :param resize_to: resize an image to an expected size
    :return: a GIF file
    """
    # Get the number of slices along the specified axis
    num_slices = img_data.shape[axis]

    # Create a temporary directory to store the slice images
    temp_dir = tempfile.mkdtemp()

    # Create a list to store the filenames of the slice images
    slice_images = []

    for i in range(num_slices):
        plt.figure()
        if axis == 0:
            plt.imshow(img_data[i, :, :], cmap='gray')
        elif axis == 1:
            plt.imshow(img_data[:, i, :], cmap='gray')
        elif axis == 2:
            plt.imshow(img_data[:, :, i], cmap='gray')
        else:
            raise ValueError("Axis must be 0, 1, or 2!!!")
        plt.axis('off')

        # Save the slice image
        slice_filename = os.path.join(temp_dir, f"slice_{i:03d}.png")
        plt.savefig(slice_filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Append the filename to the list
        slice_images.append(slice_filename)

    # Create the GIF
    gif_path = os.path.join(temp_dir, "output.gif")
    with imageio.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:
        for slice_filename in slice_images:
            image = imageio.v2.imread(slice_filename)
            if resize_to:
                image = PILImage.fromarray(image).resize(resize_to, PILImage.Resampling.LANCZOS)
                image = np.array(image)
            writer.append_data(image)

    # Display the GIF
    display(Image(filename=gif_path))

    # Clean up temporary slice images
    shutil.rmtree(temp_dir)

def create_gif_from_slices_v2(img_data, duration=40, file_name="array.gif"):
    """
    Create a GIF file
    :param file_name: name for a GIF file
    :param duration: duration is the number of milliseconds between frames
    :param img_data: an array
    :return: a GIF file
    """
    imgs = img_data.astype(np.uint32)
    imgs = [PILImage.fromarray(img) for img in imgs]  # PILImage.fromarray(): creates an image memory from an array/object(e.g, Image.open()
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save(file_name, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def process_csv_label_file(csv_file_path, data_dir):
    """
    Process .csv files

    :param csv_file_path: directory containing .csv files
    :return: list of (id,age)
    """
    csv_file = pd.read_csv(csv_file_path)
    check_nan = csv_file['AGE'].isnull()
    count_nan = csv_file['AGE'].isnull().sum()
    drop_nan = csv_file.dropna()

    # print(csv_file)
    # print('CHECK column!!!')
    # print(check_nan)
    # print('COUNT NaN!!!') # NaN: Not a Number
    # print(count_nan)
    # print('DROP NaN!!!')
    # print(drop_nan)
    # print('COUNT after drop NaN!!!')
    # print(drop_nan['AGE'].count())

    ## Check if the file name in the csv file appears in the data folder
    samples = [sample for sample in os.listdir(data_dir) if sample.startswith('IXI')]
    samples_no_filter = [sample for sample in os.listdir(data_dir)]
    find_diff_files = [item for item in samples_no_filter if item not in samples]

    # print(f"Number of files containing 'IXI' character in the IXI-T1 folder: {len(samples)}")
    # print("Different files: ", find_diff_files)

    ## Change format of the ID column to 3 digits "000"
    change_format_to_3_digits = drop_nan['IXI_ID'].astype(str).str.zfill(3)

    # print(drop_nan['IXI_ID'].dtype) # dtype: int64
    # print(change_format_to_3_digits.dtype) # dtype: object

    change_format_to_3_digits = [id for id in change_format_to_3_digits]
    # print(type(change_format_to_3_digits))

    new_df = pd.DataFrame(drop_nan)
    new_df['IXI_ID_3_digits'] = change_format_to_3_digits
    # print(f'Number of columns of dataframe: {len(new_df.columns)}')

    new_df_reordered = new_df.iloc[:, [12,0,11]]
    new_df_reordered_contain_id_age = new_df_reordered.iloc[:,[0,2]]
    # print('CHANGE FORMAT TO 3 DIGITS!!!')
    # print(new_df)
    # print(new_df_reordered)
    # print(new_df_reordered_contain_id_age)

    ## Check histogram of the IXI_ID_3_digits column --> Fail!
    # id_column_hist = new_df_reordered_contain_id_age['IXI_ID_3_digits'].hist()
    # print(id_column_hist)

    ## Check duplicate value in the column -> OK!
    # check_column = not new_df_reordered_contain_id_age['IXI_ID_3_digits'].is_unique
    # check_column = new_df_reordered_contain_id_age['IXI_ID_3_digits'].duplicated().any()
    # print("No duplication: ", check_column)
    # print(len(new_df_reordered_contain_id_age['AGE']))

    ## Exporting ID and AGE to a csv file -> OK!
    # new_df_reordered_contain_id_age.to_csv('IXI-ID-AGE.csv')

    ## Create tuple for containing id and age
    id_age_list = []
    for i in range(new_df_reordered_contain_id_age.shape[0]):
        id_age_row = new_df_reordered_contain_id_age.iloc[i]
        id_age_list.append((id_age_row.iloc[0], id_age_row.iloc[1]))

    # print(id_age_list)
    # print(id_age_list[0])
    # print(type(id_age_list[0][0])) # dtype: 'str'
    # print(type(id_age_list[0][1])) # dtype: 'np.float64'

    return id_age_list

def label_file_name(data_dir, id_age_list):
    """
    Label file names with its age, respectively

    :param data_dir: MRI data directory
    :param id_age_list: list of tupes between ID and AGE
    :return: list of tuples ('file_name',id,age), labeled file name and its age respectively
    """
    samples = [sample for sample in os.listdir(data_dir) if sample.startswith('IXI')]
    label_file_list = []
    for id_age_idx in range(len(id_age_list)):
        for sample_idx in range(len(samples)):
            if "IXI"+id_age_list[id_age_idx][0] in samples[sample_idx]:
                label_file_list.append((samples[sample_idx], id_age_list[id_age_idx][0], id_age_list[id_age_idx][1]))

    return label_file_list

def check_duplicate_file_name(data_dir, id_age_list):
    id_list = []
    age_list = []
    for idx in range(len(id_age_list)):
        id_list.append(id_age_list[idx][0])
        age_list.append(id_age_list[idx][1])

    duplicate_list = []
    for idx in id_list:
        if (id_list.count(idx) > 1) and (idx not in duplicate_list):
            duplicate_list.append(idx)
    # print("Count of duplicate elements in the list: ", len(duplicate_list))
    # print("Number of ID: ", len(id_list))
    # print("Value of duplicate elements in the list: ", duplicate_list)

    samples = [sample for sample in os.listdir(data_dir) if sample.startswith('IXI')]
    samples_Guys = [sample for sample in os.listdir(data_dir) if "Guys" in sample]
    samples_HH = [sample for sample in os.listdir(data_dir) if "HH" in sample]
    samples_IOP = [sample for sample in os.listdir(data_dir) if "IOP" in sample]

    # print("Number of Guys: ",len(samples_Guys))
    # print("Number of HH: ",len(samples_HH))
    # print("Number of IOP: ", len(samples_IOP))
    # print("Number of files in folder: ", len(samples_HH)+len(samples_Guys)+len(samples_IOP))

    filtered_id_list = []
    for id_age_idx in range(len(id_age_list)):
        if id_age_list[id_age_idx][0] not in duplicate_list:
                filtered_id_list.append(id_age_list[id_age_idx])

    print("No of removing duplicate ID: ", len(filtered_id_list))
    # print(label_file_list)
    # print(type(id_list[0]))
    # print(type(duplicate_list[0]))
    # print(type(id_age_list[0][0]))


class process_mri_data:
    def __init__(self, data_dir, id_age_list):
        self.data_dir = data_dir
        self.samples = [sample for sample in os.listdir(data_dir) if sample.startswith('IXI')]

    def __len__(self):
        return f"Length of samples in dataset: {len(self.samples)}"

    def __getitem__(self, item_idx):
        sample_name = self.samples[item_idx]
        sample_dir = os.path.join(self.data_dir, sample_name)

        ## Load MRI modalities and labels from csv files,
        ## '.nii.gz': Neuroimaging Informatics Technology Initiative (NIfTI)
        ## t1_path = os.path.join(sample_dir, sample_name)

        ## Get frame data from MRI images
        t1_data = nib.load(sample_dir).get_fdata()

        return t1_data


if __name__ == "__main__":
    ## Setup current directory
    setup_current_dir()

    ## Check GPU for use
    check_gpu()

    ## Check a middle slice of an MRI image
    """
    file_path = '../../1. Datasets/2. IXI/IXI-T1/IXI002-Guys-0828-T1.nii.gz'
    sample_path = '../Sample Images/Test'
    print_middle_mri_slice(file_path, sample_path)
    """

    ## Check 'listdir'
    """
    data_dir = '../../1. Datasets/2. IXI/Test'
    # print(os.listdir(data_dir)) # output: ['IXI524-HH-2412-T1.nii.gz', 'IXI269-Guys-0839-T1.nii.gz',...]
    # samples = [sample for sample in os.listdir(data_dir) if sample.startswith('IXI')] # output: ['IXI012-HH-1211-T1.nii.gz',...]
    samples = [sample for sample in os.listdir(data_dir)] # output: ['IXI012-HH-1211-T1.nii.gz',...,..., '002-Guys-0828-T1.nii (Copy).gz',...] 
    print(samples)
    """

    ## Check class "process_mri_data" in the part of "Loading T1 MRI images"
    """
    data_dir = '../../1. Datasets/2. IXI/Test'
    processed_data = process_mri_data(data_dir)
    # print(processed_data[0].shape)
    # import matplotlib.image
    # matplotlib.image.imsave('test_image.jpg',processed_data[0][:,:,74])
    """

    ## Check function which create a GIF from a 3D array
    """
    data_dir = '../../1. Datasets/2. IXI/Test'
    processed_data = process_mri_data(data_dir)
    # create_gif_from_slices(processed_data[0]) # Can run, but don't create a GIF file -> Fail!
    # imgs = processed_data[0].astype(np.uint16)
    # imgs = [PILImage.fromarray(img) for img in imgs] # PILImage.fromarray(): creates an image memory from an array/object(e.g, Image.open()
    # # duration is the number of milliseconds between frames; this is 40 frames per second
    # imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=1, loop=0)
    img_data = processed_data[0]
    create_gif_from_slices_v2(img_data) # Test OK!
    """

    ## Check function "process_csv_file()"
    """
    file_dir = '../../1. Datasets/2. IXI/IXI.csv'
    data_dir = '../../1. Datasets/2. IXI/IXI-T1'
    arr = process_csv_label_file(file_dir, data_dir)
    print(arr)
    """

    ## Check function "label_file_name()"
    """
    file_dir = '../../1. Datasets/2. IXI/IXI.csv'
    data_dir = '../../1. Datasets/2. IXI/IXI-T1'
    id_age_list = process_csv_label_file(file_dir, data_dir)
    labeled_file_name_list = label_file_name(data_dir, id_age_list)
    print(labeled_file_name_list)
    print(len(labeled_file_name_list))
    """

    ## Check function "check_ixi_file_name()"
    file_dir = '../../1. Datasets/2. IXI/IXI.csv'
    data_dir = '../../1. Datasets/2. IXI/IXI-T1'
    id_age_list = process_csv_label_file(file_dir, data_dir)
    check_duplicate_file_name(data_dir, id_age_list)