import cv2
import numpy as np
import os
import pickle
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

dataset_paths_real = {
    '2000': '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_dataset',
    '500': '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_dataset'
}

dataset_paths_fake = {
    '2000': '/content/drive/MyDrive/farzi_git_sayan/Fake Notes/2000',
    '500': '/content/drive/MyDrive/farzi_git_sayan/Fake Notes/500'
}

feature_paths = {
    '2000': [
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 1',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 2',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 3',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 4',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 5',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 6',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/2000_Features Dataset/Feature 7'
    ],
    '500': [
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 1',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 2',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 3',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 4',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 5',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 6',
        '/content/drive/MyDrive/farzi_git_sayan/Dataset/500_Features Dataset/Feature 7'
    ]
}

def load_image(image_path):
    """Load an image from the specified path."""
    return cv2.imread(image_path)

def preprocess_image(image):
    """Convert image to grayscale, resize, and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (300, 300))  # Resize to a fixed size
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    return blurred

def extract_features(image):
    """Extract ORB features from the image."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    """Match features between two sets of descriptors."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches

def compute_similarity(image1, image2):
    """Compute similarity score between two images based on feature matches."""
    gray1 = preprocess_image(image1)
    gray2 = preprocess_image(image2)
    kp1, des1 = extract_features(gray1)
    kp2, des2 = extract_features(gray2)
    matches = match_features(des1, des2)
    similarity_score = len(matches)
    return similarity_score

def extract_features_from_images(image_paths, label_map):
    """Extract features and labels from a list of image paths."""
    features = []
    labels = []
    for image_path in image_paths:
        image = load_image(image_path)
        gray = preprocess_image(image)
        kp, des = extract_features(gray)
        if des is not None:
            feature_vector = np.mean(des, axis=0)  # Averaging descriptors
            features.append(feature_vector)
            label = label_map.get(os.path.basename(image_path).split('_')[0], 0)
            labels.append(label)
    return np.array(features), np.array(labels)

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate a Random Forest model."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return clf

def save_model(clf, model_path):
    """Save the trained model to a file."""
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (SSIM) between two images."""
    return ssim(img1, img2, data_range=img2.max() - img2.min())

def test_feature(dataset_name, image_size):
    """Test feature extraction and template matching for a given dataset and image size."""
    dataset_paths = feature_paths[dataset_name]
    test_img_path = os.path.join(dataset_paths_real[dataset_name], f'{image_size}_s1.jpg')
    
    score_set_list = []
    best_extracted_img_list = []
    avg_ssim_list = []

    for j in range(len(dataset_paths)):
        print(f'ANALYSIS OF FEATURE {j+1}')
        score_set = []
        max_score = -1
        max_score_img = None

        # Process exactly 6 templates
        for i in range(1, 7):
            print(f'---> Template {i} :')
            template_path = os.path.join(dataset_paths[j], f'{i}.jpg')
            template_img = cv2.imread(template_path)
            if template_img is None:
                print(f"Error: Unable to load template image from {template_path}")
                continue
            
            template_img_gray = preprocess_image(template_img)

            # Load and preprocess the test image
            test_img = cv2.imread(test_img_path)
            if test_img is None:
                print(f"Error: Unable to load test image from {test_img_path}")
                continue

            test_img_gray = preprocess_image(test_img)
            
            # Perform feature matching
            kp1, des1 = extract_features(template_img_gray)
            kp2, des2 = extract_features(test_img_gray)
            if des1 is None or des2 is None:
                continue
            matches = match_features(des1, des2)
            
            # Draw matches
            matches_img = cv2.drawMatches(template_img_gray, kp1, test_img_gray, kp2, matches[:20], None, flags=2)
            plt.imshow(matches_img)
            plt.show()

            # Calculate average SSIM
            avg_ssim = calculate_ssim(template_img_gray, test_img_gray)
            avg_ssim_list.append(avg_ssim)
            score_set.append(avg_ssim)
            if avg_ssim > max_score:
                max_score = avg_ssim
                max_score_img = template_path

        best_extracted_img_list.append(max_score_img)
        score_set_list.append(score_set)
        print(f'Best template for feature {j+1} is {max_score_img} with score: {max_score:.2f}')

    return best_extracted_img_list, score_set_list, avg_ssim_list

def main():
    real_image_paths = [os.path.join(path, f) for path in dataset_paths_real.values() for f in os.listdir(path) if f.endswith('.jpg')]
    fake_image_paths = [os.path.join(path, f) for path in dataset_paths_fake.values() for f in os.listdir(path) if f.endswith('.jpg')]
    all_image_paths = real_image_paths + fake_image_paths
    labels = [1] * len(real_image_paths) + [0] * len(fake_image_paths)  # 1 for real, 0 for fake
    dummy_label_map = {os.path.basename(path).split('_')[0]: 1 for path in all_image_paths}
    features, labels = extract_features_from_images(all_image_paths, label_map=dummy_label_map)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)
    clf = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    model_path = '/content/drive/MyDrive/farzi_git_sayan/farzimodel.pkl'
    save_model(clf, model_path)
    for dataset in ['2000', '500']:
        print(f'\nTesting feature extraction for dataset {dataset}:')
        for size in ['2000', '500']:  # Test both sizes if needed
            best_imgs, score_sets, avg_ssims = test_feature(dataset, size)
            print(f"Image size: {size}")
            print("Best images:", best_imgs)
            print("Score sets:", score_sets)
            print("Average SSIMs:", avg_ssims)

if __name__ == "__main__":
    main()
