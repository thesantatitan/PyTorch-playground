import numpy as np
from PIL import Image, ImageDraw
import random
import os

def generate_circle_image(width, height, min_radius, max_radius, min_width, max_width):
    # Create a black background
    image = Image.new('RGB', (width, height), color='black')
    draw = ImageDraw.Draw(image)

    # Generate random circle parameters
    radius = random.uniform(min_radius, max_radius)
    circle_width = random.uniform(min_width, max_width)
    
    # Calculate circle position to ensure it's fully within the image
    x = width//2
    y = height//2

    # Draw the circle
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                 outline='white', width=int(circle_width))

    return image

def main():
    # Get user input
    width = int(input("Enter image width: "))
    height = int(input("Enter image height: "))
    num_images = int(input("Enter number of images to generate: "))

    # Set circle parameters
    min_radius = min(width, height) * 0.1
    max_radius = min(width, height) * 0.4
    min_width = 1
    max_width = min(width, height) * 0.2

    # Create output folder if it doesn't exist
    if not os.path.exists('circle_images'):
        os.makedirs('circle_images')

    # Generate and save images
    for i in range(num_images):
        img = generate_circle_image(width, height, min_radius, max_radius, min_width, max_width)
        img.save(f'circle_images/circle_{i+1}.png')
        print(f"Generated image {i+1}/{num_images}")

    print("All images generated successfully.")

if __name__ == "__main__":
    main()
