import cv2
import numpy as np

image = cv2.imread('../img/FPS.jpg', cv2.IMREAD_COLOR)

if image is None:
    print("Error: Unable to read the image. Please check the file path.")
else:

    cv2.imshow('OriginalImage', image)

    image_float = image.astype(np.float32) / 255.0

    blurred = cv2.blur(image_float, (5, 5))

    contrast = image_float - blurred

    contrast_normalized = np.zeros_like(contrast)
    for c in range(contrast.shape[2]):
        min_contrast = np.min(contrast[:, :, c])
        max_contrast = np.max(contrast[:, :, c])
        contrast_normalized[:, :, c] = (contrast[:, :, c] - min_contrast) / (max_contrast - min_contrast)

    noise_prob_map = np.zeros_like(contrast_normalized)
    noise_intensity_map = np.zeros_like(contrast_normalized)
    for c in range(contrast_normalized.shape[2]):
        noise_prob_map[:, :, c] = np.where(contrast_normalized[:, :, c] > 0.5, 0.05, 0.02)
        noise_intensity_map[:, :, c] = np.where(contrast_normalized[:, :, c] > 0.5, 0.05, 0.02)

    noisy_image = image_float.copy()
    noise_mask = np.random.rand(image_float.shape[0], image_float.shape[1])

    for c in range(image_float.shape[2]):
        for i in range(image_float.shape[0]):
            for j in range(image_float.shape[1]):
                prob = noise_prob_map[i, j, c]
                intensity = noise_intensity_map[i, j, c]
                if noise_mask[i, j] < prob * intensity:
                    noisy_image[i, j, c] = 1.0
                elif noise_mask[i, j] > 1 - prob * intensity:
                    noisy_image[i, j, c] = 0.0

    noisy_image = (noisy_image * 255).astype(np.uint8)
    cv2.imwrite('NoisyImage.jpg', noisy_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()