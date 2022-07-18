# My notes and observations as I train

Found a link that talks about scaling the learning rate on multi gpu by NVIDIA [here](https://docs.nvidia.com/clara/tlt-mi_archive/clara-train-sdk-v2.0/nvmidl/appendix/training_with_multiple_gpus.html)

## Notes from Ross Wightman:

"Have you ever seen TPU final accuracy being lower than that of GPU's?"
> Yup, happens all the time if you use pure bfloat16. I get pretty good reproductions with float32 activations, but bfloat16 causes issues with reductions like avg pool, means, std-dev, etc. As well as things like cross entropy loss, softmax, etc

> If this happens it will look like you're hitting an accuracy ceiling of sorts during training that is much lower than where it would be on GPU or float32. Sometimes that ceiling stays stuck, sometimes it hits it and spirals down to the ground.

> It can also bounce off the ground and repeat, but it will never hit the accuracy it could or should.

## NLP Example

If the weights are never updated/optimizer wasn't stepped/we did not train, accuracy stays at 0.31617 and f1 at 0.0