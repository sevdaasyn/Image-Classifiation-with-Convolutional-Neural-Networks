# Image-Classifiation-with-Convolutional-Neural-Networks
Transfer learning by using CNNs and how to train a network for a specific problem and pre-trained VGG-16 model usage


### FILES

* part_1.py (Image Representation)
* part_2.py (Fine-tuning)
* part_3.py 



### COMMON FUNCTIONS

` eval_model(vgg, criterion)
	used in part_2
train_model(model, criterion, optimizer, scheduler, num_epochs)
	used in part_2, part_3
calculate_svm(model)
	used in part_1, part_3
calculate_classbased_accuracies(feat_train, feat_classes_train, feat_test, feat_classes_test, clf)
	used in part_1, part_3 `


### DEPENDENCIES

` pip install torch
...
import torch (part_1.py,part_2.py, part_3.py)
import torch.nn as nn (part_1.py,part_2.py, part_3.py)
import torch.optim as optim (part_2.py, part_3.py)
from torch.optim import lr_scheduler (part_2.py, part_3.py)
from torchvision import models (part_1.py,part_2.py, part_3.py)
from torch.autograd import Variable (part_1.py,part_2.py, part_3.py) `

` pip install torchvision
...
from torchvision import datasets, models, transforms (part_1.py,part_2.py, part_3.py) `

` pip install sklearn
...
from sklearn.svm import LinearSVC (part_1.py, part_3.py) ` 
