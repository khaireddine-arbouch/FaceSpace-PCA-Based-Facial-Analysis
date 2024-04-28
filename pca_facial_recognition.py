import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from skimage import color
from skimage import transform
import zipfile
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from joblib import Parallel, delayed

# Define constants
IMAGE_SIZE = (128, 128)
NUM_COMPONENTS = 100
NUM_FACES_TO_VISUALIZE = 5
NUM_EIGENFACES_TO_VISUALIZE = 10

# Load the zip file
zip_file = zipfile.ZipFile(r"C:\Users\khair\OneDrive\Desktop\CelebFaces Attributes (CelebA) Dataset\archive.zip")

# Get the list of image files within the zip file
image_files = [f for f in zip_file.namelist() if f.startswith('img_align_celeba/img_align_celeba/') and f.endswith('.jpg')]

# Define a function to read an image from the zip file
def read_image_from_zip(zip_file, image_file):
    """
    Read an image from the zip file and convert it to a NumPy array.
    
    Parameters:
    zip_file (ZipFile): The zip file containing the images.
    image_file (str): The name of the image file within the zip file.
    
    Returns:
    img (numpy.ndarray): The image as a NumPy array.
    """
    with zip_file.open(image_file) as f:
        img = Image.open(f)
        img = np.array(img)
        return img

# Read the first image to get the image size
img = read_image_from_zip(zip_file, image_files[0])
image_size = img.shape[:2]

# Convert images to grayscale and transform into 1D vectors
image_vectors = []
for image_file in tqdm(image_files, desc="Loading and processing images"):
    img = read_image_from_zip(zip_file, image_file)
    img = Image.fromarray(img)  # convert to PIL image
    img = img.resize(IMAGE_SIZE)  # downsample the image to 128x128
    img = np.array(img)  # convert back to NumPy array
    img = color.rgb2gray(img)
    img = img.reshape(-1)  # flatten the image into a 1D vector
    image_vectors.append(img)

image_vectors = np.array(image_vectors)

# Center the data
mean_vector = np.mean(image_vectors, axis=0)
centered_data = image_vectors - mean_vector

# Calculate covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Perform PCA
pca = PCA(n_components=NUM_COMPONENTS)
pca_data = pca.fit_transform(centered_data)

# Sort eigenvalues and eigenvectors in descending order
eigenvalues = pca.explained_variance_ratio_
eigenvectors = pca.components_

# Visualize the eigenfaces
plt.figure(figsize=(15, 5))
for i in range(NUM_EIGENFACES_TO_VISUALIZE):
    plt.subplot(2, NUM_EIGENFACES_TO_VISUALIZE // 2, i + 1)
    eigenface = eigenvectors[i].reshape(image_size)
    plt.imshow(eigenface, cmap='gray')
    plt.title(f'Eigenface {i+1}')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Reconstruct faces using PCA
reconstructed_faces = pca.inverse_transform(pca_data) + mean_vector

# Visualize original and reconstructed faces
plt.figure(figsize=(15, 5))
for i in range(NUM_FACES_TO_VISUALIZE):
    plt.subplot(2, NUM_FACES_TO_VISUALIZE, i + 1)
    original_face = image_vectors[i].reshape(image_size)
    plt.imshow(original_face, cmap='gray')
    plt.title('Original Face')
    plt.axis('off')

    plt.subplot(2, NUM_FACES_TO_VISUALIZE, i + 1 + NUM_FACES_TO_VISUALIZE)
    reconstructed_face = reconstructed_faces[i].reshape(image_size)
    plt.imshow(reconstructed_face, cmap='gray')
    plt.title('Reconstructed Face')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Define a function to find the closest match
def find_closest_match(target_face, faces_data, num_components):
    """
    Find the closest match to a target face in the dataset.
    
    Parameters:
    target_face (numpy.ndarray): The target face as a NumPy array.
    faces_data (numpy.ndarray): The dataset of faces as a NumPy array.
    num_components (int): The number of principal components to use.
    
    Returns:
    closest_index (int): The index of the closest match in the dataset.
    """
    # Project target face onto principal components
    target_projected = np.dot(target_face - mean_vector, eigenvectors[:num_components])
    
    # Project all faces in the dataset onto principal components
    dataset_projected = np.dot(faces_data - mean_vector, eigenvectors[:num_components])
    
    # Calculate distances between target face and all faces in the dataset
    distances = np.linalg.norm(dataset_projected - target_projected, axis=1)
    
    # Find the index of the closest match
    closest_index = np.argmin(distances)
    
    return closest_index

# Choose a target face (e.g., the first face in the dataset)
target_face_index = 0
target_face = centered_data[target_face_index]

# Find the closest match
closest_index = find_closest_match(target_face, centered_data, NUM_COMPONENTS)

# Display the closest match
closest_match_face = image_vectors[closest_index].reshape(image_size)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(target_face.reshape(image_size), cmap='gray')
plt.title('Target Face')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(closest_match_face, cmap='gray')
plt.title('Closest Match')
plt.axis('off')

plt.tight_layout()
plt.show()

# Define a function to generate random faces
def generate_random_faces(num_faces_to_generate, num_components):
    """
    Generate random faces using the principal components.
    
    Parameters:
    num_faces_to_generate (int): The number of random faces to generate.
    num_components (int): The number of principal components to use.
    
    Returns:
    random_faces (numpy.ndarray): The generated random faces as a NumPy array.
    """
    # Generate random coefficients for principal components
    random_coefficients = np.random.randn(num_faces_to_generate, num_components)
    
    # Project random coefficients back into the original data space
    random_faces_projected = np.dot(random_coefficients, eigenvectors[:, :num_components].T)
    
    # Reconstruct random faces from projected data
    random_faces = mean_vector + random_faces_projected
    
    return random_faces

# Generate random faces
num_faces_to_generate = 5
random_faces = generate_random_faces(num_faces_to_generate, NUM_COMPONENTS)

# Visualize the generated faces
plt.figure(figsize=(15, 5))
for i in range(num_faces_to_generate):
    plt.subplot(1, num_faces_to_generate, i + 1)
    generated_face = random_faces[i].reshape(image_size)
    plt.imshow(generated_face, cmap='gray')
    plt.title(f'Generated Face {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Evaluate reconstruction accuracy using mean squared error (MSE)
mse_reconstruction = np.mean((image_vectors - reconstructed_faces)**2)
print(f'Mean Squared Error (MSE) for reconstruction: {mse_reconstruction}')