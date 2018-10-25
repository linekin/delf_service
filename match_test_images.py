import os
import tablib
from pathlib import Path
from tqdm import tqdm
from delf_features import load_features, extract, Feature, match, match_features
from dotenv import load_dotenv

load_dotenv()


def match_test_images():
    features_dir = Path(os.getenv('FEATURES_DIR'))
    haystack = load_features(features_dir)

    images_paths = [str(image_path) for image_path in Path('test_images').iterdir()]
    needles = extract(images_paths)

    res = []
    for i, needle in tqdm(enumerate(needles)):
        matched = match_features(needle, haystack)
        tup = (images_paths[i], matched['love.delf'])
        tqdm.write(str(tup))
        res.append(tup)

    write_csv(res)


def write_csv(res):
    res.sort(key=lambda tup: tup[1], reverse=True)
    data = tablib.Dataset()
    for tup in res:
        data.append(tup)
    with open('test_images_result.csv', 'w') as f:
        f.write(data.csv)


if __name__ == '__main__':
    match_test_images()
