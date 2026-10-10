Stride Ahead is an AI-powered pedestrian assistance prototype that turns camera footage into clear warnings about potential outdoor hazards. It combines computer vision with a fine-tuned vision-language model to detect obstacles, understand their context, and explain potential risks along a person’s walking path.

The system identifies where objects appear, highlights potential hazards, and provides directional warnings such as “Person ahead; possible collision hazard.” Visual overlays and spoken alerts help communicate what needs attention and why.

Our goal is to make information about the surrounding environment more accessible, supporting pedestrian awareness through understandable, contextual guidance.

The fine tuned and trained weights for this project are:

Gemma 4 12B fine-tuned: We fine-tuned this vision-language model to analyze camera frames alongside RF-DETR’s detections. It considers the scene and the person’s intended walking path to describe potential hazards. For example, explaining that a person ahead may obstruct the sidewalk.
RF-DETR Medium trained: We trained this model to recognize outdoor objects such as pedestrians, vehicles, poles, and curbs. It returns labels and bounding boxes that show where each object appears in the camera image.
Both model available on Hugging Face:
https://huggingface.co/Atlearia/Gemma_4_finetuned
