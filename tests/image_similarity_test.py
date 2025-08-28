import torch
import numpy as np
import os
import argparse
from models.backbone_attention import Backbone
import torchvision
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from utils.data_transforms import ZeroMeanNormalize, ZeroOneNormalize, LetterBoxResize
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import normalize
import cv2


def extract_feature(model, image_path, transforms, device, use_letterbox=False):
    """Extract feature from a single image"""
    try:
        if use_letterbox:
            data = cv2.imread(image_path)
        else:
            data = read_image(image_path, mode=ImageReadMode.RGB)
            data = data.to(device)
        
        data = transforms(data).unsqueeze(dim=0).to(device)
        
        with torch.no_grad():
            feature = model(data)
            feature = normalize(feature, dim=1)
            feature = feature.cpu().detach().numpy()
        
        return feature
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def compare_images(image1_path, image2_path, model_path, similarity_threshold=0.8):
    """Compare similarity between two images"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    backbone = Backbone(out_dimension=1024, model_name="resnet50", pretrained=False)
    model, _, _ = backbone.build_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Setup transforms
    use_letterbox = False
    val_transforms_list = [
        LetterBoxResize(dst_size=(224, 224)) if use_letterbox else torchvision.transforms.Resize(size=(224, 224)),
        ZeroOneNormalize(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    val_transforms = torchvision.transforms.Compose(val_transforms_list)
    
    # Extract features
    print(f"Processing {image1_path}...")
    feature1 = extract_feature(model, image1_path, val_transforms, device, use_letterbox)
    
    print(f"Processing {image2_path}...")
    feature2 = extract_feature(model, image2_path, val_transforms, device, use_letterbox)
    
    if feature1 is None or feature2 is None:
        print("Failed to extract features from one or both images")
        return None
    
    # Calculate similarity
    similarity = cosine_similarity(feature1, feature2)[0][0]
    distance = 1.0 - similarity
    
    print(f"\n--- Similarity Analysis ---")
    print(f"Image 1: {os.path.basename(image1_path)}")
    print(f"Image 2: {os.path.basename(image2_path)}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Cosine Distance: {distance:.4f}")
    print(f"Similarity Threshold: {similarity_threshold}")
    
    # Determine if images are similar
    if similarity >= similarity_threshold:
        print(f"✓ Images are SIMILAR (similarity: {similarity:.4f} >= {similarity_threshold})")
        result = "similar"
    else:
        print(f"✗ Images are NOT SIMILAR (similarity: {similarity:.4f} < {similarity_threshold})")
        result = "not_similar"
    
    return {
        'similarity': similarity,
        'distance': distance,
        'result': result,
        'threshold': similarity_threshold
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare similarity between two images')
    parser.add_argument('--image1', required=True, help='Path to first image')
    parser.add_argument('--image2', required=True, help='Path to second image')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold (default: 0.8)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found: {args.image1}")
        exit(1)
    
    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found: {args.image2}")
        exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        exit(1)
    
    # Compare images
    result = compare_images(args.image1, args.image2, args.model, args.threshold)
    
    if result:
        print(f"\n--- Final Result ---")
        print(f"Result: {result['result'].upper()}")


# python tests/image_similarity_test.py --image1 datasets/imgs/cn/0.jpg --image2 datasets/imgs/cn/0T.jpg --model ./logs/model-acc-299-0.9473-0.9869-0.9526.pth --threshold 0.8 