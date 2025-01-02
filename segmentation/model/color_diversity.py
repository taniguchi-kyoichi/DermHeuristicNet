import cv2
import numpy as np
from sklearn.cluster import KMeans


class ColorDiversitySegmentation:
    def __init__(self, n_clusters=5):
        """
        Initialize the color diversity segmentation module.

        Parameters:
        -----------
        n_clusters : int
            Number of color clusters to use for diversity analysis
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

    def _extract_colors(self, image, mask):
        """
        Extract colors from the lesion area defined by the mask.

        Parameters:
        -----------
        image : ndarray
            Original RGB image
        mask : ndarray
            Binary mask of the lesion area

        Returns:
        --------
        ndarray
            Array of LAB colors from the lesion area, shaped as (n_pixels, 3)
        """
        try:
            # Convert to LAB color space
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Convert mask to single channel if it's RGB
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Create binary mask
            binary_mask = mask > 127

            # Extract lesion pixels
            lesion_pixels = lab_image[binary_mask]

            if lesion_pixels.size == 0:
                return None

            # Ensure correct shape for clustering (n_samples, n_features)
            if len(lesion_pixels.shape) == 1:
                # If we got a 1D array, reshape it properly
                n_pixels = lesion_pixels.size // 3  # Divide by 3 for L,a,b channels
                lesion_pixels = lesion_pixels.reshape(n_pixels, 3)

            return lesion_pixels

        except Exception as e:
            print(f"Error in _extract_colors: {str(e)}")
            return None

    def _compute_color_distances(self, colors, centers):
        """
        Compute distances from each pixel to all cluster centers.

        Parameters:
        -----------
        colors : ndarray
            Array of LAB colors
        centers : ndarray
            Cluster center colors

        Returns:
        --------
        ndarray
            Array of distances to each cluster center
        """
        try:
            distances = np.zeros((colors.shape[0], centers.shape[0]))
            for i, center in enumerate(centers):
                distances[:, i] = np.sqrt(np.sum((colors - center) ** 2, axis=1))
            return distances

        except Exception as e:
            print(f"Error in _compute_color_distances: {str(e)}")
            return None

    def _create_diversity_map(self, image_shape, mask, colors, distances):
        """
        Create a diversity map based on color distances.

        Parameters:
        -----------
        image_shape : tuple
            Shape of the original image
        mask : ndarray
            Binary mask of the lesion area
        colors : ndarray
            Array of LAB colors
        distances : ndarray
            Array of distances to cluster centers

        Returns:
        --------
        ndarray
            Color diversity map
        """
        try:
            # Initialize diversity map
            diversity_map = np.zeros(image_shape[:2], dtype=np.float32)

            # Convert mask to single channel if it's RGB
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

            # Get lesion pixel coordinates
            lesion_coords = np.where(mask > 127)

            if len(lesion_coords[0]) == 0:
                return np.zeros(image_shape[:2], dtype=np.uint8)

            # Calculate diversity scores
            min_distances = np.min(distances, axis=1)
            max_dist = np.max(min_distances) + 1e-6  # Avoid division by zero
            diversity_scores = min_distances / max_dist

            # Assign scores to the map
            diversity_map[lesion_coords] = diversity_scores

            # Normalize to 0-255 range
            diversity_map = (diversity_map * 255).astype(np.uint8)

            return diversity_map

        except Exception as e:
            print(f"Error in _create_diversity_map: {str(e)}")
            return np.zeros(image_shape[:2], dtype=np.uint8)

    def generate_diversity_map(self, image, mask):
        """
        Generate a color diversity map for the given image and mask.

        Parameters:
        -----------
        image : ndarray
            Original RGB image
        mask : ndarray
            Binary segmentation mask

        Returns:
        --------
        ndarray
            Color diversity map (RGB format)
        """
        try:
            # Ensure input image is RGB
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Extract colors from lesion area
            lesion_colors = self._extract_colors(image, mask)

            if lesion_colors is None or len(lesion_colors) < self.n_clusters:
                # Return blank map if not enough pixels
                return cv2.cvtColor(
                    np.zeros(image.shape[:2], dtype=np.uint8),
                    cv2.COLOR_GRAY2RGB
                )

            # Fit KMeans to lesion colors
            self.kmeans.fit(lesion_colors)

            # Compute distances to cluster centers
            distances = self._compute_color_distances(
                lesion_colors,
                self.kmeans.cluster_centers_
            )

            if distances is None:
                return cv2.cvtColor(
                    np.zeros(image.shape[:2], dtype=np.uint8),
                    cv2.COLOR_GRAY2RGB
                )

            # Generate diversity map
            diversity_map = self._create_diversity_map(
                image.shape,
                mask,
                lesion_colors,
                distances
            )

            # Apply median blur to reduce noise
            diversity_map = cv2.medianBlur(diversity_map, 5)

            # Convert to RGB format
            diversity_map_rgb = cv2.cvtColor(diversity_map, cv2.COLOR_GRAY2RGB)

            return diversity_map_rgb

        except Exception as e:
            print(f"Error generating diversity map: {str(e)}")
            return np.zeros_like(image)