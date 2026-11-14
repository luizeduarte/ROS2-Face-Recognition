"""Build or extend a face library offline from a folder of photos.

Expected layout (one sub-folder per person, any number of photos each):

    people/
      ada/      1.jpg  2.jpg
      grace/    a.png

Usage:
    ros2 run face_recognition_ros enroll_from_images people/ [--library PATH]

Photos with zero or several faces are skipped, since it would be ambiguous which face
belongs to the label.
"""

import argparse
import sys
from pathlib import Path

import cv2

from face_recognition_ros.face_library import FaceLibrary
from face_recognition_ros.models import FaceDetector, FaceEmbedder

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_LIBRARY = '~/.local/share/face_recognition_ros/library.npz'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', type=Path, help='folder with one sub-folder per person')
    parser.add_argument('--library', default=DEFAULT_LIBRARY, help='library file (.npz)')
    args = parser.parse_args(argv)

    detector, embedder = FaceDetector(), FaceEmbedder()
    library = FaceLibrary.load_or_create(args.library)
    added = 0

    for person in sorted(p for p in args.folder.iterdir() if p.is_dir()):
        for photo in sorted(person.iterdir()):
            if photo.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(photo))
            if image is None:
                print(f'skip  {photo}: unreadable')
                continue
            faces = detector.detect(image)
            if len(faces) != 1:
                print(f'skip  {photo}: found {len(faces)} faces, expected 1')
                continue
            library.add(person.name, embedder.embed(image, faces[0]))
            added += 1
            print(f'added {photo} -> {person.name}')

    library.save(args.library)
    print(f'\n{added} embeddings added; library now has {len(library)} embeddings for '
          f'{len(library.identities)} identities: {", ".join(library.identities)}')
    return 0 if added else 1


if __name__ == '__main__':
    sys.exit(main())
