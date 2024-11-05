from llm_axe.models import OllamaChat
from llm_axe import ObjectDetectorAgent

# Load a multimodal LLM capable of detecting objects in an image
# llava:7b works great.
llm = OllamaChat(model="llavca:7b")
detector = ObjectDetectorAgent(llm, llm)


# It will detect objects according to our criteria
# If we want all objects, we can use the detection_criteria="Detect all objects in the image."
resp = detector.detect(images=["20210509115556.jpg"], detection_criteria="Detect all objects in the image.")
print(resp)
