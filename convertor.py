import argparse
import math

import cv2
import numpy as np


def diamond_grid_mosaic(input_path, output_path, tile_size=40):
    print(f"Converting {input_path} to {output_path} with tile size {tile_size}.")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    h, w, _ = img.shape
    canvas = np.zeros_like(img)

    half = tile_size // 2

    for row, y in enumerate(range(0, h + tile_size, half)):
        # stagger every other row
        x_offset = 0 if row % 2 == 0 else half

        for x in range(x_offset, w + tile_size, tile_size):
            pts = np.array([
                (x, y - half),  # top
                (x + half, y),  # right
                (x, y + half),  # bottom
                (x - half, y)  # left
            ], np.int32)

            # Clip to image boundaries
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts, 1)

            mean_color = cv2.mean(img, mask=mask)[:3]
            cv2.fillConvexPoly(canvas, pts, mean_color)

    cv2.imwrite(output_path, canvas)
    print(f"Diamond mosaic saved at {output_path}")


def triangle_grid_mosaic_equ(input_path, output_path, tile_size):
    print(f"Converting {input_path} to {output_path} with tile size {tile_size}.")
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {input_path}")

    h, w, _ = img.shape

    # Start from the original image so uncovered pixels aren't black
    canvas = img.copy()

    # Use rounded height to avoid cumulative truncation error
    tri_height = int(round((math.sqrt(3) / 2) * tile_size))

    # Expand the loops so triangles outside the frame get clipped in,
    # ensuring full coverage at the borders
    y_start = -tri_height
    y_end = h + 2 * tri_height
    x_start = -tile_size
    x_end = w + tile_size

    for row, y in enumerate(range(y_start, y_end, tri_height)):
        x_offset = 0 if row % 2 == 0 else tile_size // 2

        for x in range(x_start + x_offset, x_end, tile_size):
            # Upward triangle
            pts_up = np.array([
                (x, y),
                (x + tile_size // 2, y + tri_height),
                (x - tile_size // 2, y + tri_height)
            ], dtype=np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, pts_up, 255)  # 255 mask for cv2.mean

            if mask.any():
                mean_color = cv2.mean(img, mask=mask)[:3]
                cv2.fillConvexPoly(canvas, pts_up, mean_color)

            # Downward triangle
            pts_down = np.array([
                (x, y + 2 * tri_height),
                (x + tile_size // 2, y + tri_height),
                (x - tile_size // 2, y + tri_height)
            ], dtype=np.int32)

            mask[:] = 0
            cv2.fillConvexPoly(mask, pts_down, 255)

            if mask.any():
                mean_color = cv2.mean(img, mask=mask)[:3]
                cv2.fillConvexPoly(canvas, pts_down, mean_color)

    cv2.imwrite(output_path, canvas)
    print(f"Equilateral triangle mosaic saved at {output_path}")


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
parser.add_argument("--input_file", default="test_org.jpg", type=str, help="Path to the input file.", required=False)
parser.add_argument("--out_put", default="100_test_output.jpg", type=str, help="Path to the input file.",
                    required=False)
parser.add_argument("--tile_size", default=100, type=int, help="Path to the input file.", required=False)
args = parser.parse_args()

triangle_grid_mosaic_equ(args.input_file, args.out_put, args.tile_size)
# diamond_grid_mosaic(args.input_file, args.out_put, args.tile_size)
# triangle_grid_mosaic(args.input_file, args.out_put, args.tile_size)
