from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_path = r"C:\Users\gabri\Desktop\All_Portfolio_Projects\Image_Captioning_Ai\test_images\3_91f02503-53e6-4c36-8452-128bf8d51788.jpg"
image = Image.open(img_path).convert("RGB")

# Unconditional captioning: no text prompt
inputs = processor(images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("Caption:", processor.decode(outputs[0], skip_special_tokens=True))

# Conditional captioning: the model continues your prompt
inputs = processor(images=image, text="a photography of", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print("Prompted:", processor.decode(outputs[0], skip_special_tokens=True))