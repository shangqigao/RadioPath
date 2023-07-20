from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig
from tiatoolbox.models.architecture import CNNBackbone

model = CNNBackbone("resnet50")
extractor = DeepFeatureExtractor(batch_size=32, model=model, num_loader_workers=4)
output_map_list = extractor.predict()
