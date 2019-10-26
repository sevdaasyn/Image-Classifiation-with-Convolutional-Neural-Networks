# Image-Classifiation-with-Convolutional-Neural-Networks
Transfer learning by using CNNs and how to train a network for a specific problem and pre-trained VGG-16 model usage


### FILES

* part_1.py (Image Representation)
* part_2.py (Fine-tuning)
* part_3.py 



### COMMON FUNCTIONS

` eval_model(vgg, criterion) `<br/>
`	used in part_2 ` <br/>
` train_model(model, criterion, optimizer, scheduler, num_epochs) ` <br/>
`	used in part_2, part_3 ` <br/>
` calculate_svm(model) ` <br/>
`	used in part_1, part_3 ` <br/>
` calculate_classbased_accuracies(feat_train, feat_classes_train, feat_test, feat_classes_test, clf) ` <br/>
`	used in part_1, part_3 ` <br/>


### DEPENDENCIES

` pip install torch ` <br/>
` ... ` <br/>
` import torch (part_1.py,part_2.py, part_3.py) ` <br/>
` import torch.nn as nn (part_1.py,part_2.py, part_3.py) ` <br/>
` import torch.optim as optim (part_2.py, part_3.py) ` <br/>
` from torch.optim import lr_scheduler (part_2.py, part_3.py) ` <br/>
` from torchvision import models (part_1.py,part_2.py, part_3.py) ` <br/>
` from torch.autograd import Variable (part_1.py,part_2.py, part_3.py)  ` <br/>

` pip install torchvision ` <br/>
` ...` <br/>
` from torchvision import datasets, models, transforms (part_1.py,part_2.py, part_3.py) ` <br/>

` pip install sklearn ` <br/>
` ... ` <br/>
` from sklearn.svm import LinearSVC (part_1.py, part_3.py) ` <br/>
