
# 17.01.24
Training, WhichfaceisReal Dataset
classifier
```PY

vgg16 = models.vgg16(pretrained=True)

#deactivate training for pretrained layers
for param in vgg16.features.parameters():
    param.requires_grad = False #friere gewichte ein 

#Parameter Block
vgg16.avgpool = nn.AdaptiveAvgPool2d(output_size=(2,2))  # featuremap auf kartengröße 1*1 setzen (alt war 7*7 : zu viele werte/gewichte) aber immer noch 512 veschiedene karten
# featuremap ist trz mehrdimensional mit 512 unterschiedlichen sachen
vgg16.classifier = nn.Sequential(  # sequential = nacheinander
        nn.Flatten(),  # serialisieren auf 1 dimension?
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),  # ein output neuron : binäre klassifikation in cat or dog
        nn.Sigmoid())
loss_fn = nn.BCELoss()

# define loss function and optimizer BCE=BinaryCrossEntropy
criterion = nn.BCELoss() #cross entropy is better for binary classificators
optimizer = optim.SGD(vgg16.parameters(), lr=0.0001, momentum=0.5,)

#fix this later
#summary(vgg16, input_size=(3, 224, 224))
#print("")

```