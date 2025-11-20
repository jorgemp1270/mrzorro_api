import base64

def encode_image_to_base64(image_path, output_file):
    try:
        # Read the image file in binary mode
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()

        # Encode the image data to base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')

        # Save the base64 string to a text file
        with open(output_file, 'w') as txt_file:
            txt_file.write(base64_encoded)

        print(f"Image encoded and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: {image_path} not found")
    except Exception as e:
        print(f"Error: {e}")

# Encode img.png and save to base64_image.txt
encode_image_to_base64('img.jpeg', 'base64_image.txt')