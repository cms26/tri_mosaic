import argparse
import cv2
import numpy as np


def triangle_grid_mosaic(input_path, output_path, tile_size):
    print(f"Converting {input_path} to {output_path} with tile size {tile_size}.")
    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    h, w, _ = img.shape
    canvas = np.zeros_like(img)

    height_index = range(0, h, tile_size)
    # Loop through the grid
    for y in height_index:
        width_index = range(0, w, tile_size)
        for x in width_index:
            # Define triangle points (two per square cell)
            p1 = (x, y)
            p2 = (x + tile_size, y)
            p3 = (x, y + tile_size)
            p4 = (x + tile_size, y + tile_size)
            # print('x', x)
            # First triangle (top-left)
            if x + tile_size < w and y + tile_size < h:
                pts1 = np.array([p1, p2, p3], np.int32)
                mask1 = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask1, pts1, 1)
                color1 = cv2.mean(img, mask=mask1)[:3]
                cv2.fillConvexPoly(canvas, pts1, color1)

                # Second triangle (bottom-right)
                pts2 = np.array([p2, p3, p4], np.int32)
                mask2 = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask2, pts2, 1)
                color2 = cv2.mean(img, mask=mask2)[:3]
                cv2.fillConvexPoly(canvas, pts2, color2)

    # Save result
    cv2.imwrite(output_path, canvas)
    print(f"Grid-based triangular mosaic saved at {output_path}")


parser = argparse.ArgumentParser(description="Converts file to low poly tri file")
parser.add_argument("--input_file", default="test_org3.png", type=str, help="Path to the input file.", required=False)
parser.add_argument("--out_put", default="45_test_output.png", type=str, help="Path to the input file.", required=False)
parser.add_argument("--tile_size", default=55, type=int, help="Path to the input file.", required=False)
args = parser.parse_args()

triangle_grid_mosaic(args.input_file, args.out_put, args.tile_size)
