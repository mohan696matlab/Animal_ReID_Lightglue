import os
import sys
from pathlib import Path
import base64
import torch
import faiss
import numpy as np
from PIL import Image
import json
from dino_feature_extractor import FeatureExtractor
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

class LogReIDInference:
    def __init__(self,
                 database_folder,
                 top_k=5,
                 min_num_matches=20,
                 device=None):
        
        self.faiss_index_path = Path(database_folder)/'faiss_index.idx'
        self.features_superpoint_folder = Path(database_folder)/'features_superpoint'
        self.index_to_path_dict_path = Path(database_folder)/'index_to_path_dict.json'

        self.top_k = top_k
        self.min_num_matches = min_num_matches
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self._setup()
    
    def _setup(self):
        print("Loading FAISS index...")
        self.index = faiss.read_index(rf"{self.faiss_index_path}")
        
        print("Loading models...")
        self.feature_extractor = FeatureExtractor().to(self.device)
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", filter_threshold=0.8, n_layers=9).eval().to(self.device)

        print("Loading database image indexes...")
        with open(self.index_to_path_dict_path, 'r') as f:
            self.index_to_path_dict = json.load(f)
            
    # Helper method inside the class
    def _read_image_bytes(self, image_path):
        try:
            with open(image_path, 'rb') as img_file:
                # Encode as base64 so it's JSON serializable
                return base64.b64encode(img_file.read()).decode('utf-8')
        except Exception as e:
            return f"Error reading image: {e}"


    def find_num_matches_btn_two_images(self, feats0, feats1):
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        return matches01['matches'][0].shape[0]

    def find_match_from_database(self, query_features, database_features):
        matched_objects = []
        for idx, db_feat in enumerate(database_features):
            num_matches = self.find_num_matches_btn_two_images(query_features, db_feat)
            if num_matches >= self.min_num_matches:
                matched_objects.append({
                    'matched': True,
                    'match_num': num_matches,
                    'image': self._read_image_bytes(self.index_to_path_dict[str(idx)]),
                })
                
                
        if len(matched_objects)==0:
            matched_objects.append({
                    'matched': 'No matches',
                    'match_num': 'No matches',
                    'image': None,
                })
                
        # Sort matched_objects based on 'match_num' in descending order
        matched_objects.sort(key=lambda x: x['match_num'], reverse=True)
        
        return matched_objects[0]

    def run_inference(self, query_image_path):
        # print(f"\nRunning inference for: {query_image_path}\n")

        # Step 1: Global retrieval using FAISS
        query_image = Image.open(query_image_path)
        query_features = self.feature_extractor(query_image).detach().cpu().numpy()
        D, I = self.index.search(query_features, k=self.top_k)
        
        # Step 2: Local feature matching using LightGlue
        # I is typically shape (1, top_k), so flatten it
        selected_images_keypoints = []
        for idx in I.flatten():
            if idx>=0:
                pt_file = self.features_superpoint_folder / f"{idx}.pt" 
                features = torch.load(pt_file)
                selected_images_keypoints.append(features)
                

    # Step 2: Local feature matching using LightGlue
        query_image_keypoints = self.extractor.extract(load_image(query_image_path, resize=512).to(self.device))

        matched_objects = self.find_match_from_database(query_image_keypoints, selected_images_keypoints)
        
        return matched_objects


# =================== USAGE EXAMPLE ===================

if __name__ == "__main__":

    from pathlib import Path

    database_folder = Path(__file__).parent / "VECTOR_DATABASE"
    
    
    inference_engine = LogReIDInference(
        database_folder=database_folder,
    )

    # Example: Run inference on first database test image
    query_image_path = "/home/mohan/Log_ReID/IMAGE_DATABASE/FB220034467_00583_R1.jpg"
    matched_objects = inference_engine.run_inference(query_image_path)

    print("\nMatched Objects:\n", matched_objects)
