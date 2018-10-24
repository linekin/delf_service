import os
from pathlib import Path
from tqdm import tqdm
from delf_features import load_features, extract, Feature, match, match_features
from dotenv import load_dotenv

load_dotenv()


def match_test_images():
    features_dir = Path(os.getenv('FEATURES_DIR'))
    haystack = load_features(features_dir)

    res = {}
    for img_path in tqdm(list(Path('test_images').iterdir())[:10]):
        locations, descriptors = extract(img_path)
        matched = match_features(Feature(locations, descriptors), haystack)
        res[str(img_path)] = matched

    assert len(res) > 0


if __name__ == '__main__':
    match_test_images()
