Ghanshyam Y.

12:10 PM, Sep 12

Profile photo
Hi Everyone! Anyone started working on the project?

Profile photo
2 Replies Latest reply 3 months ago

Christopher I.

3:54 AM, Sep 13

Profile photo
I recently received confirmation that we will not have continued access to course material post-graduation. If that troubles you, I strongly suggest writing mailto:mailto:dsnd-support@udacity.com or using their feedback system to request that they do not change that part of the service model.

Maia A.

2:12 PM, Sep 17

Profile photo
In the analyzing student data lab, the train_nn function is written from scratch. Is there any good scikit learn function that does this for you?

Profile photo
Profile photo
2 Replies Latest reply 3 months ago

Sam W.

9:23 AM, Sep 18

Profile photo
Hi, I have started the project, however every time I try to run my model, which is built on a pretrained VGG11 model, I get the error: RuntimeError: cuda runtime error (59) : device-side assert triggered at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCTensorMath.cu:26

when running the loss.backward() step.

I even went back to the lesson code to see if there was something I was doing wrong in my training step, but I get the same error when using the training steps from the lesson.

Sam W.

9:24 AM, Sep 18

Profile photo
I'm wondering if there is something I'm doing wrong in the data prep steps, but I don't see what it would be.

Sam W.

12:05 PM, Sep 18

Profile photo
I think I found the problem to this 'device-side assert' issue that I was having in the back-propagation step.

The issue seemed to be that I had only 2 outputs on my classifier, when there should have been as many outputs as classes in the training dataset.

Christopher I.

4:17 AM, Sep 19

Profile photo
In the very last pytorch lesson, transfer learning, I must be making an error in my transforms, because I keep getting this error on the model training section:

RuntimeError: Given input size: (1024x3x3). Calculated output size: (1024x-3x-3). Output size is too small at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THNN/generic/SpatialAveragePooling.c:64
Christopher I.

4:18 AM, Sep 19

Profile photo
Not sure what I'm doing wrong, here are my training transforms:

transforms.Compose([transforms.Resize(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(100),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
Profile photo
Profile photo
3 Replies Latest reply 3 months ago

Christopher I.

4:19 AM, Sep 19

Profile photo
too bad Udacity doesn't give us Slack-like functionality for editing posts, I could clean that up and turn it into a thread, but, I can't

Sam W.

8:28 AM, Sep 19

Profile photo
Hi, Christopher. I saw some errors similar to those. I think what is going on here is the cropped image size does not fit the input size of the network. The image classifiers given by torchvision have an input size of 224x224, so that's ultimately what you want to end up with out of the transforms.

I would change the resize to something larger than 224, like 256 or so, then RandomizeCrop to 224, so you get most of the image, but crop off the edges, like you were cutting the crust off of a slice of bread. You want larger images to be close to 224, so you resize first, then trim some off the edges.

Profile photo
1 Reply Latest reply 3 months ago

Sam W.

8:36 AM, Sep 19

Profile photo
Actually this documentation says they need to be shape (3xHxW) and they need to be at least 224, so I guess you could go larger. https://pytorch.org/docs/stable/torchvision/models.html

Christopher I.

1:33 AM, Sep 20

Profile photo
Thank you! Another question--in Image Classifier Part 1, it looks like we're supposed to load all data (train, test, valid) with one ImageFolder object? How do we simultaneously load all three folders, apply 3 sets of transforms, and create 3 dataloaders with as few function calls as possible? the way the comments in the cell are written, it looks like they expect us to do this without creating ImageFolder objects and dataloaders for each folder separately.

Profile photo
Profile photo
2 Replies Latest reply 3 months ago

Christopher I.

2:38 AM, Sep 20

Profile photo
I'm also confused why the notebook, in the validation area, refers to testloader rather than validloader. My understanding is that we have 3 datasests, train, valid, test, and that we use train to train, valid to check the trained model, and test for prediction and final evaluation. so, shouldn't all this code in the validation section be referring to the valid_dataset?

Profile photo
1 Reply Latest reply 3 months ago

Sam W.

9:29 AM, Sep 20

Profile photo
I'm not sure if I'm doing it right, because I haven't submitted anything for review, but I actually created three different loaders, and commented out the incomplete assignments that were in the notebook.

I also agree with you on not understanding why we are not using the validation loader in that part. I really haven't finished yet, so I may go back and change that, but so far I didn't use the validation data. I only trained, and used the test set to test.

Sam W.

9:34 AM, Sep 20

Profile photo
I think once I get everything basically done, and the part 2 scripts mostly working, I'm going to go through more models and hyperparameters, and let it run for a while to see if I can get a little bit better score. I think that may be when I introduce the validation data officially. I'm basically trying to get everything off the ground first, then once the process works end-to-end, I'm going to test it and finish it.

Profile photo
Profile photo
3 Replies Latest reply 3 months ago

Christopher I.

9:34 PM, Sep 20

Profile photo
I followed the instructions to save whenever activating/deactivating the GPU, and yet, much of my work is gone

Christopher I.

10:51 PM, Sep 20

Profile photo
Also, another issue. I resize the images as instructed, keeping the aspect ratio, but the result for an image is a size of 256x204, which is smaller than the need 224x224

Profile photo
Profile photo
6 Replies Latest reply 3 months ago

Christopher I.

11:24 PM, Sep 20

Profile photo
Okay, I just did a ton more work, and after saving, I disabled the GPU, and when the notebook reloaded, all of my work was gone again! WTF!!!!!!

Christopher I.

11:24 PM, Sep 20

Profile photo
It reset all the back to the same checkpoint that loaded earlier today, the first time I noticed missing work. So, for some reason, it keeps going back to that checkpoint.

Profile photo
2 Replies Latest reply 3 months ago

Ghanshyam Y.

1:14 AM, Sep 21

Profile photo
You can use conditonals to check for the shorter side and then multiply/divide the other one by the aspect ratio

Profile photo
1 Reply Latest reply 3 months ago

Vemula Dilip K.

1:49 AM, Sep 21

Profile photo
How the given block is working in the below given code kindly explain me is_correct_string = 'Yes' if output == correct_output else 'No'

import pandas as pd

TODO: Set weight1, weight2, and bias
weight1 = 7 weight2 = 6 bias = 18

DON'T CHANGE ANYTHING BELOW
Inputs and outputs
test_inputs = [(0, 0), (0, 1), (1, 0), (1, 1)] correct_outputs = [False, False, False, True] outputs = []

Generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs): linear_combination = weight1 * test_input[0] + weight2 * test_input[1] + bias output = int(linear_combination >= 0) is_correct_string = 'Yes' if output == correct_output else 'No' outputs.append([test_input[0], test_input[1], linear_combination, output, is_correct_string])

Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No']) output_frame = pd.DataFrame(outputs, columns=['Input 1', ' Input 2', ' Linear Combination', ' Activation Output', ' Is Correct']) if not num_wrong: print('Nice! You got it all correct.\n') else: print('You got {} wrong. Keep trying!\n'.format(num_wrong)) print(output_frame.to_string(index=False))

Christopher I.

1:56 AM, Sep 21

Profile photo
I understand from the class notebook how to save and load a model, but what is different about reconstructing a model when it is built on a pretrained model? What should I be saving in the checkpoint in this case?

Is it possible to store and load the features and classifier as well, or do I need to rebuild them first separately and then load the state dict?

Christopher I.

3:58 AM, Sep 21

Profile photo
Yet another question--the instructions say to normalize the PIL images in process image, but, the means we are supposed to subtract from the color channels do not make sense. In the flower images, color values range fro 0-255, so how could [0.485, 0.456, 0.406] possibly be the means of all colors for all images? Same for the sd. I've have tried using these values to normalize the colors, which of course is not working. I think I must be misunderstanding something.

Profile photo
1 Reply Latest reply 3 months ago

Sam W.

10:45 AM, Sep 21

Profile photo
Hey, Christopher, on the normalization thing, I had success by taking each of the 3 color channels and dividing the whole array by 255 (to get a number between 0 and 1), then subtracting that by the mean and dividing the whole thing by the std deviation.

So, in pseudo python: normalized_array = [3,224,224] for channel in img_array: normalized_array = ((img_array[channel]/255)-means[channel])/std_dev[channel]

This works because each color channel pixel has some number between 0 and 255 that we want to make between 0 and 1.

I also found that pytorch seems to have simpler saving and loading process, which is recommended by the documentation. I have been using it, and haven't had any issues. My work hasn't been checked, though, so it may not be correct, according to the requirements of the assignment.

https://pytorch.org/docs/stable/notes/serialization.html#recommend-saving-models

Samuel Alexander R.

9:06 PM, Sep 22

Profile photo
I am not understanding this part: The sigmoid function is defined as sigmoid(x) = 1/(1+e-x). If the score is defined by 4x1 + 5x2 - 9 = score, then which of the following points has exactly a 50% probability of being blue or red? (Choose all that are correct.) Can someone explain?

Sam W.

12:24 PM, Sep 23

Profile photo
Hi, Samuel. I think I can help a little bit, but might have to review the course material to give a better answer. The linear function that defines the score is passed into the sigmoid function as x. The sigmoid function returns a value between -1 and 1, where 0 is the value it returns when the point has a 50% probability of being either class.

Profile photo
1 Reply Latest reply 2 months ago

Tahsin Ashraf C.

11:35 AM, Sep 24

Profile photo
Hi all, I'm doing the image classifier project and i've run into a problem. When I'm training my NN, my accuracy is increasing but by validation loss is also increasing and my training loss goes close to 0 within the first few steps.

Can't seem to figure out why this is was, but my initial suspicion is that the model is overfitting. I tried increasing the dropout probability to tackle this but no luck.

Can anyone help me out here? Thanks in advance!

Profile photo
Profile photo
9 Replies Latest reply 3 months ago

Tahsin Ashraf C.

4:50 PM, Sep 24

Profile photo
Hi all, ran into some trouble. I'm doing the image classifier project. I'm using densenet169 I have successfully trained, tested and saved my checkpoint. I can also load the checkpoint and run it on the test dataset. However, when I try to make prediction, I'm getting the following error: RuntimeError: size mismatch, m1: [1 x 6656], m2: [1664 x 500] at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCTensorMathBlas.cu:249

My suspicion is that the mismatch could be due to how i am preporcessing the image as the model works perfectly fine on the test, validation and training datasets.

Here are my process_image and predict functions:

```def process_image(image):

# TODO: Process a PIL image for use in a PyTorch model

# Resizing
image = image.resize((256, 256))
width, height = image.size
new_width = 224
new_height = 224
left = (width - new_width)/2
top = (height - new_height)/2
right = (width + new_width)/2
bottom = (height + new_height)/2
image.crop((left, top, right, bottom))

# Converting color channels from 0-255 to 0-1
image = np.array(image)
image = image/255

# Normalizing
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = (image - mean) / std

# Reordering
image = image.transpose((2, 0, 1))

return image```
```def predict(image_path, model, topk=5):

# TODO: Implement the code to predict the class from an image file
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

image = Image.open(image_path)
image = process_image(image)
image = torch.from_numpy(image).type(torch.FloatTensor)
image = image.to(device)
image = image.unsqueeze(0)
output = model.forward(image.float())
ps = torch.exp(output)
return prob, indx = ps.topk(topk)```
Running the following throws the mismatch error: predict('/home/workspace/aipnd-project/flowers/test/100/image_07896.jpg', model, topk=5)

Any suggestions for how to go about this would be greatly appreciated. Thanks in advance!

Profile photo
Profile photo
2 Replies Latest reply 3 months ago

Tahsin Ashraf C.

12:52 PM, Sep 25

Profile photo
Is there a particular time when I can get a hold of a mentor? I've been struggling for a while now looking for help across all platforms, with absolutely no response

Profile photo
1 Reply Latest reply 3 months ago

Himanshu M.

Mentor

3:31 PM, Sep 25

Profile photo
Hi Everyone !! I just got the access to mentorship portal. Feel free to ask the questions and tag me.

Himanshu M.

Mentor

3:32 PM, Sep 25

Profile photo
It is requested to provide the complete code and error-trace so that we can help debug faster.

Profile photo
14 Replies Latest reply 3 months ago

Christopher I.

12:44 AM, Sep 26

Profile photo
How can we determine the output size of a pretrained features model that can change depending on user input for part 2? There must be some method or parameters dict we can access to get this info, but I can't find one.

Profile photo
Profile photo
3 Replies Latest reply 3 months ago

Tahsin Ashraf C.

1:00 PM, Sep 26

Profile photo
Hi guys, I just started Part 2. The ting is i'm not very well acquainted to command line/python scripting. Yes, I know the basics of terminal but I'm not very confident. what do you think the best course of action would be for me?

Profile photo
Profile photo
4 Replies Latest reply 3 months ago

Christopher I.

3:23 AM, Sep 27

Profile photo
The latest stumbling block: RuntimeError: Found 0 files in subfolders of: C:\Users\#####\Desktop\DSND\DSND_Term1\projects\p2_image_classifier\classifier\data_dir\train Supported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif

I have a complete path to some sample training images, but datasets.ImageFolder gives this error.

Profile photo
Profile photo
8 Replies Latest reply 3 months ago

Tahsin Ashraf C.

4:40 AM, Sep 27

Profile photo
Do I have to include the flowers folder in my submission zip? If yes where can i get it from?

Profile photo
Profile photo
3 Replies Latest reply 3 months ago

Christopher I.

4:52 AM, Sep 27

Profile photo
One more question I don't yet have the answer to: so far, for rebuilding a model from a checkpoint, I still have to download or load the pretrained model, then replace the classifier, and only after that can I load the state_dict. Is there a way to load a saved, trained model without having to first construct the untrained model? Are there efficient ways to save everything in the checkpoint after training?

Profile photo
Profile photo
8 Replies Latest reply 3 months ago

Vemula Dilip K.

6:02 AM, Sep 27

Profile photo
kindly suggest me how can I install keras on my system

Profile photo
Profile photo
5 Replies Latest reply 3 months ago

Christopher I.

1:39 AM, Sep 28

Profile photo
anyone have any thoughts on the best way to pass parser arguments across all the different functions for part 2? using *args and *kwargs for every function, and setting the arguments inside each function seems like a lot of work.

Profile photo
Profile photo
4 Replies Latest reply 3 months ago

Himanshu M.

Mentor

9:43 PM, Sep 30

Profile photo
Hi everyone, how are your projects coming along ? Can I help anyone?

Profile photo
Profile photo
13 Replies Latest reply 3 months ago

Li Z.

7:44 PM, Oct 1

Profile photo
Hey guys! I have a non-technical question: does the material improve after supervised learning? Because I have found the teaching for that aspect to be incredibly disappointing, I have never felt so let down in my long history of using this service.

Profile photo
Profile photo
5 Replies Latest reply 2 months ago

Jennifer W.

5:45 PM, Oct 2

Profile photo
Hi, I'm working on the last section of the PyTorch lesson, Transfer Learning. Regardless of what model I use I get an accuracy of 49% or 50%. I even get the same when I run the solution notebook. Does anyone have an idea why this is happening?

Profile photo
Profile photo
3 Replies Latest reply 2 months ago

Himanshu M.

Mentor

7:58 AM, Oct 3

Profile photo
Hi everyone, can I help anyone today?

Per V.

9:42 AM, Oct 3

Profile photo
Hi Himanshu! I was wondering about the Rubric. In the jupyter notebook it is stated that all imports are to be made on the first cell. After we save the model and want to load it again, why not assume we have no imports, no previous info whatsoever? That is imports need to made again similar to the top cell to load the model and predict from it there after. Do you agree?

Per V.

9:42 AM, Oct 3

Profile photo
Sorry -> For the jupyter nootbook it is stated

Himanshu M.

Mentor

10:31 AM, Oct 3

Profile photo
@PerV The whole point of loading model part in notebook is to do a demo run of checkpoint loading functionality. In daily like of researcher/data scientist, we don't start to write code in python file, but we generally integrate it module by module, by running each part separately in notebook first, that's what the entire excercise is about. There are a few good practices which should be followed to keep the notebook clean and that's what is being enforced in the part 1 of the project.

Profile photo
1 Reply Latest reply 3 months ago

Alazar K.

10:37 AM, Oct 3

Profile photo
Hey guys sorry if that has been asked before , I am getting out of memory error when training the model. Has anyone run into this issue before?RuntimeError: cuda runtime error (2) : out of memory at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCStorage.cu:58

Alazar K.

10:54 AM, Oct 3

Profile photo
@ Himanshu M.

Alazar K.

10:54 AM, Oct 3

Profile photo
RuntimeError: cuda runtime error (2) : out of memory at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCStorage.cu:58

Profile photo
Profile photo
Profile photo
8 Replies Latest reply 2 months ago

Liberto S.

10:20 AM, Oct 4

Profile photo
Hi, I am stuck on Transfer Learning Solution.

Liberto S.

10:20 AM, Oct 4

Profile photo
RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'

Profile photo
1 Reply Latest reply 2 months ago

Liberto S.

10:22 AM, Oct 4

Profile photo
What is happening here? I thought a solution code should have no error.

Liberto S.

10:25 AM, Oct 4

Profile photo
I just applied a solution offered from Knowledge section, but only got 50% accuracy. The text mentioned that I should get better than 95% accuracy easily.

Alazar K.

12:27 PM, Oct 4

Profile photo
@HimanshuM what is the impact of using ('output', nn.LogSoftmax(dim=1)) vs not using it when computing the loss = criterion(outputs, labels) during training. Let's assume labels is from 0 to 9. But the output from the classifier will be 10 floats with each between 0 and 1 and sum up to 1.

Alazar K.

12:34 PM, Oct 4

Profile photo
@LibertoS make sure you do images.to('cuda') and labels.to('cuda'). Certainly that solution code needs to be fixed.

Himanshu M.

Mentor

12:23 AM, Oct 5

Profile photo
@LibertoS the issue is that, your model is running on gpu, but the images are not, to fix that you also need to move the images on the gpu as answered by @AlazarK above. Hope this helps :)

Profile photo
1 Reply Latest reply 2 months ago

Himanshu M.

Mentor

12:37 AM, Oct 5

Profile photo
@AlazarK In that case our Loss function is Negative Log Loss, the logits are expected to be in range 0-1, of you won't use logsoftmax, the NLL function will not be calculated correctly.

Himanshu M.

Mentor

12:38 AM, Oct 5

Profile photo
sum(label * log(prediction))
Himanshu M.

Mentor

12:39 AM, Oct 5

Profile photo
this is how muliticlass classification formula looks like, as you can see the input should be log(softmax(prediction)) -> logit

Himanshu M.

Mentor

12:39 AM, Oct 5

Profile photo
I how this clears things a bit :)

Vemula Dilip K.

3:35 AM, Oct 5

Profile photo
import numpy as np

import torch

ImportError Traceback (most recent call last) <ipython-input-6-466bbead58b5> in <module>() 1 import numpy as np ----> 2 import torch

~\Anaconda3\lib\site-packages\torch_init_.py in <module>() 74 pass 75 ---> 76 from torch.C import * 77 78 _all += [name for name in dir(_C)

ImportError: DLL load failed: The specified module could not be found.

I am Unable to import torch, kindly suggest me ,how can i do it?

Profile photo
Profile photo
4 Replies Latest reply 2 months ago

Maia A.

12:56 PM, Oct 5

Profile photo
Are neural networks only useful for image identification? Or do they have other common applications?

Profile photo
Profile photo
Profile photo
3 Replies Latest reply 2 months ago

Per V.

3:16 PM, Oct 5

Profile photo
Hi. I am wondering how the residuals in the ResNet et al family get fed back into the model as described by the documentation/papers. Is through the forward function? If so pls explain in detail pls. Pls also point out the exact code snippet or example where it takes place so I can examine that part of the code in detail. Thx in adv!

Maia A.

3:20 PM, Oct 5

Profile photo
How do we know what input size the model is expecting? I keep getting "RuntimeError: input has less dimensions than expected" on the line "output = model.forward(image)". I need tips on debugging this issue?

Profile photo
Profile photo
Profile photo
20 Replies Latest reply 2 months ago

xun Y.

5:29 AM, Oct 8

Profile photo
Hi, I have a question about the lab of lesson one in Deep learning part (Analyzing Student Data). To my understanding, the two-layer neural network has one input layer and one output layer. It is the same a logistic regression. Thus the 'error_term_formula' should return (y-output) but not (y-output)*sigmoid_prim(x).

11:26 AM, Oct 8

Profile photo
@HimanshuM This question might have been asked before - regarding loading a saved model. So I have trained a vgg16 model ( modified a the classifier network - adding few more layers instead of just modifying the output layer.) Now, after saving the state_dict , I am getting error while trying to load. Here is how I am reloading it: vgg16 = models.vgg16() vgg16.load_state_dict(torch.load(path)) and here the error I am getting Traceback (most recent call last): File "predict.py", line 20, in <module> reload_model(args.checkpoint) File "/home/workspace/aipnd-project/dl_model.py", line 84, in reload_model vgg16.load_state_dict(torch.load(path)) File "/opt/conda/lib/python3.6/site-packages/torch/nn/modules/module.py", line 721, in load_state_dict self.__class__.__name__, "\n\t".join(error_msgs))) RuntimeError: Error(s) in loading state_dict for VGG: Missing key(s) in state_dict: "classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias", "classifier.6.weight", "classifier.6.bias". Unexpected key(s) in state_dict: "classifier.fc1.weight", "classifier.fc1.bias", "classifier.fc2.weight", "classifier.fc2.bias", "classifier.fc3.weight", "classifier.fc3.bias".

Profile photo
Profile photo
5 Replies Latest reply 2 months ago

Alazar K.

4:05 PM, Oct 8

Profile photo
Hello guys, @HimanshuM during training one of the requirements is to pass architecture as part of the command line argument. I am not sure how you can create a model give the name of the architecture ``` models.'archName'(pretrained=True)) where the archName comes from the command line argument.

Profile photo
Profile photo
11 Replies Latest reply 2 months ago

Per V.

9:58 AM, Oct 9

Profile photo
@HimanshuM Hi. I am wondering how the residuals in the ResNet et al family get fed back into the model as described by the documentation/papers. Is through the forward function? If so pls explain in detail pls. Pls also point out the exact code snippet or example where it takes place so I can examine that part of the code in detail. Thx in adv!

Profile photo
Profile photo
14 Replies Latest reply 2 months ago

Vemula Dilip K.

1:56 AM, Oct 10

Profile photo
helper.view_classify(img.view(1, 28, 28), ps) I am Unable to user helper to view the data and unable to import helper, so kindly suggest me how can i do the same working?

Profile photo
1 Reply Latest reply 2 months ago

rigoberto C.

5:26 PM, Oct 11

Profile photo
one question do we have to use pytorch for this project or can we just use keras

Profile photo
Profile photo
4 Replies Latest reply 2 months ago

Nitin C.

5:38 AM, Oct 12

Profile photo
How to save the model as a checkpoint?

Profile photo
1 Reply Latest reply 2 months ago

Alazar K.

9:40 AM, Oct 12

Profile photo
@HimanshuM I am getting the following error while reloading the model

Alazar K.

9:41 AM, Oct 12

Profile photo
RuntimeError                              Traceback (most recent call last)
<ipython-input-8-61583b35de4b> in <module>()
----> 1 epochs, model, optimizer = load_checkpoint(path)

<ipython-input-7-623b2a279119> in load_checkpoint(filepath)
      1 # TODO: Write a function that loads a checkpoint and rebuilds the model
      2 def load_checkpoint(filepath):
----> 3     state = torch.load(path)
      4     epochs = state['epoch']
      5     model = Create_model()

/opt/conda/lib/python3.6/site-packages/torch/serialization.py in load(f, map_location, pickle_module)
    301         f = open(f, 'rb')
    302     try:
--> 303         return _load(f, map_location, pickle_module)
    304     finally:
    305         if new_fd:

/opt/conda/lib/python3.6/site-packages/torch/serialization.py in _load(f, map_location, pickle_module)
    467     unpickler = pickle_module.Unpickler(f)
    468     unpickler.persistent_load = persistent_load
--> 469     result = unpickler.load()
    470
    471     deserialized_storage_keys = pickle_module.load(f)

/opt/conda/lib/python3.6/site-packages/torch/serialization.py in persistent_load(saved_id)
    435             if root_key not in deserialized_objects:
    436                 deserialized_objects[root_key] = restore_location(
--> 437                     data_type(size), location)
    438             storage = deserialized_objects[root_key]
    439             if view_metadata is not None:

/opt/conda/lib/python3.6/site-packages/torch/serialization.py in default_restore_location(storage, location)
     86 def default_restore_location(storage, location):
     87     for _, _, fn in _package_registry:
---> 88         result = fn(storage, location)
     89         if result is not None:
     90             return result

/opt/conda/lib/python3.6/site-packages/torch/serialization.py in _cuda_deserialize(obj, location)
     68     if location.startswith('cuda'):
     69         device = max(int(location[5:]), 0)
---> 70         return obj.cuda(device)
     71
     72

/opt/conda/lib/python3.6/site-packages/torch/_utils.py in _cuda(self, device, non_blocking, **kwargs)
     66         if device is None:
     67             device = -1
---> 68     with torch.cuda.device(device):
     69         if self.is_sparse:
     70             new_type = getattr(torch.cuda.sparse, self.__class__.__name__)

/opt/conda/lib/python3.6/site-packages/torch/cuda/__init__.py in __enter__(self)
    223         if self.idx is -1:
    224             return
--> 225         self.prev_idx = torch._C._cuda_getDevice()
    226         if self.prev_idx != self.idx:
    227             torch._C._cuda_setDevice(self.idx)

RuntimeError: cuda runtime error (35) : CUDA driver version is insufficient for CUDA runtime version at torch/csrc/cuda/Module.cpp:51
Profile photo
Profile photo
8 Replies Latest reply 2 months ago

Joshua T.

8:47 PM, Oct 13

Profile photo
Looks like most of you are WAY past me. I was struggling hard with backprop. This is where I finally (started) to make sense of it: https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c

Profile photo
1 Reply Latest reply 2 months ago

Samuel Alexander R.

4:20 PM, Oct 14

Profile photo
Mentors: Why do I get a 'module' object is not callable here. # TODO: Define your transforms for the training, validation, and testing sets data_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

TODO: Load the datasets with ImageFolder
image_datasets = datasets.ImageFolder(train_dir, transform=transforms)

TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, shuffle=True) image, label = next(iter(dataloaders)) helper.imshow(image[0,:]);

Samuel Alexander R.

5:13 PM, Oct 14

Profile photo
Nevermind. It was my image_dataset. It should be transform=data_transforms. Thanks if you were going to help me anyways!

Maia A.

11:27 AM, Oct 15

Profile photo
What does this error mean? RuntimeError: cuda runtime error (59) : device-side assert triggered I've seen several people ask about it in knowledge, but no responses.

Profile photo
Profile photo
4 Replies Latest reply 2 months ago

Maia A.

11:49 AM, Oct 15

Profile photo
Can someone please point me to a section in the lessons that goes over how to use validation set in training? I am confused by this instruction in the project: "Track the loss and accuracy on the validation set to determine the best hyperparameters"

Profile photo
Profile photo
31 Replies Latest reply about 2 months ago

Himanshu M.

Mentor

11:55 AM, Oct 15

Profile photo
@MaiaA Here is a code snippet to guide you, it's little old, I wrote it for pytorch 0.3, but it will good idea about how to perform validation:

for e in range(epochs):
    for images, labels in iter(trainloader):
        steps += 1
#         images.resize_(images.size()[0], 3*224*224)

        # Wrap images and labels in Variables so we can calculate gradients
        inputs = Variable(images)
        targets = Variable(labels)
        optimizer.zero_grad()

        output = model.forward(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]

        if steps % print_every == 0:
            # Model in inference mode, dropout is off
            model.eval()

            accuracy = 0
            val_loss = 0
            for ii, (images, labels) in enumerate(validloader):
#                 images = images.resize_(images.size()[0], 3*224*224)
                # Set volatile to True so we don't save the history
                inputs = Variable(images, volatile=True)
                labels = Variable(labels, volatile=True)

                output = model.forward(inputs)
                val_loss += loss_fn(output, labels).data[0]

                ## Calculating the accuracy
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output).data
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(val_loss/len(validloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))

            running_loss = 0

            # Make sure dropout is on for training
            model.train()
Profile photo
Profile photo
3 Replies Latest reply 2 months ago

Michael R.

1:42 PM, Oct 15

Profile photo
hi guys, for some reason I am getting RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 269 and 224 in dimension 2 at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/TH/generic/THTensorMath.c:3586

Profile photo
1 Reply Latest reply 2 months ago

Michael R.

1:43 PM, Oct 15

Profile photo
this is when simply trying to print this line of code

Michael R.

1:43 PM, Oct 15

Profile photo
for images, labels in dataloaders_test: print(images)

Michael R.

1:43 PM, Oct 15

Profile photo
for images, labels in dataloaders_test: print(images)

Michael R.

1:44 PM, Oct 15

Profile photo
would this be right for the transforms for the test set?

Michael R.

1:44 PM, Oct 15

Profile photo
data_test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

Profile photo
Profile photo
7 Replies Latest reply about 2 months ago

Samuel Alexander R.

6:33 PM, Oct 15

Profile photo
Why am I having a size mismatch RuntimeError? I posted my issue here: https://knowledge.udacity.com/questions/13325

Knowledge Post

I am getting a size mismatch RuntimeError, and I don't understand why.
{__('knowledge answers')}
1 ANSWER
Profile photo
Profile photo
2 Replies Latest reply 2 months ago

Michael R.

12:11 PM, Oct 16

Profile photo
I am not getting less than a 4.260 loss for some reason and I cannot seem to figure out why. Here is my transforms

[transforms.Resize(256), transforms.CenterCrop(224), # this is to reduce the noise from the data transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]

Profile photo
Profile photo
8 Replies Latest reply 2 months ago

Junying R.

9:51 AM, Oct 17

Profile photo
@HimanshuM Hi mentor, I'm confused in the first quiz in Lesson Keras. Could you tell me why the Dense of the last layer should be set "2" instead of "1"? In the example, the Dense of the output layer is "1".

Profile photo
Profile photo
21 Replies Latest reply 2 months ago

Vidhu Shekhar S.

12:06 AM, Oct 18

Profile photo
Hi Himanshu, I am geting an error while doing deep learning with pytorch - Part 7 - Loading Image Data

Vidhu Shekhar S.

12:06 AM, Oct 18

Profile photo
AttributeError Traceback (most recent call last) <ipython-input-26-d3f4fa7c2237> in <module>() 2 3 # TODO: Define transforms for the training data and testing data ----> 4 train_transforms = transforms.Compose([transforms.RandomRotation(30), 5 transforms.RandomResizedCrop(100), 6 transforms.RandomHorizontalFlip(),

AttributeError: 'Compose' object has no attribute 'Compose'

Vidhu Shekhar S.

12:07 AM, Oct 18

Profile photo
for below data Augmentaion code

Vidhu Shekhar S.

12:07 AM, Oct 18

Profile photo
data_dir = '../Cat_Dog_data'

TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(100), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])

Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms) test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32) testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

Vidhu Shekhar S.

12:10 AM, Oct 18

Profile photo
Any hint? What did I do wrong? how can I correct it?

Profile photo
Profile photo
7 Replies Latest reply 2 months ago

Nitin C.

7:46 AM, Oct 18

Profile photo
Hi!! In the second part for command line app do we have to specify a new model with hyperparameters values specified same as Train Network -> Options?

Profile photo
Profile photo
2 Replies Latest reply 2 months ago

rigoberto C.

8:34 PM, Oct 18

Profile photo
hello do we have to use the pre-trained neural network from torchvison

Profile photo
Profile photo
2 Replies Latest reply about 2 months ago

rigoberto C.

8:34 PM, Oct 18

Profile photo
or can we just build one using pytorch

Alazar K.

2:48 PM, Oct 19

Profile photo
@HimanshuM I am getting the following message when I loaded my notebook file I am almost 98% done with this project. ``` This exercise has been updated. Reset all files, code, and databases to the updated course content

Backup any progress by copying or downloading your code and files.

Any changes you have made will be overwritten. ``` Do I have reset the data and start over?

Profile photo
1 Reply Latest reply about 2 months ago

rigoberto C.

10:30 AM, Oct 20

Profile photo
hi for the train.py file can i just rewrite the code from the jupyter notebook and just add the arguments pass through the command line

Profile photo
Profile photo
2 Replies Latest reply about 2 months ago

Samuel Alexander R.

7:56 PM, Oct 20

Profile photo
How do I get the right label number instead of the tensor number. E.g. I get tensor(0) for folder 1 in the testing pictures. I would like to get a 1 so I can map it to the classess in the json document.

Profile photo
Profile photo
6 Replies Latest reply about 2 months ago

Samuel Alexander R.

9:02 PM, Oct 20

Profile photo
I just created a helper function to do it. But I wonder if there is an easier way to do it.

Wei Chun C.

10:07 PM, Oct 20

Profile photo
```data_dir = 'flowers'

TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {'train':transforms.Compose([ transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'test':transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 'valid':transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

TODO: Load the datasets with ImageFolder
image_datasets = {str(x): datasets.ImageFolder(os.path.join(data_dir, str(x)), transform=data_transforms[str(x)]) for x in [train,valid,test]}

TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle = True) for x in ['train','valid','test']}```

I keep encounter the following bug. Can anyone help me?

<ipython-input-22-20ebac3dd484> in <dictcomp>(.0)
     20
     21 # TODO: Load the datasets with ImageFolder
---> 22 image_datasets = {str(x): datasets.ImageFolder(os.path.join(data_dir, str(x)), transform=data_transforms[str(x)]) for x in [train,valid,test]}
     23
     24 # TODO: Using the image datasets and the trainforms, define the dataloaders

KeyError: 'flowers/train/'
Profile photo
Profile photo
3 Replies Latest reply about 2 months ago

Juan Alberto A.

1:16 PM, Oct 21

Profile photo
Hi, I can't save my model because there is an 'Unexpected Error' while saving. [Errno28] No space left on device. I'm trying to save the model.state_dict(), as well as the optimizer.state.dict()

Profile photo
1 Reply Latest reply about 2 months ago

Juan Alberto A.

1:17 PM, Oct 21

Profile photo
What should I do to save the model?

Michael R.

8:35 AM, Oct 22

Profile photo
anyone have a problem getting into their workspace?

Profile photo
Profile photo
Profile photo
3 Replies Latest reply about 2 months ago

Michael R.

2:58 PM, Oct 22

Profile photo
if there are any mentors present on this, please note that I am unable to access my worspace still

Michael R.

2:59 PM, Oct 22

Profile photo
no matter if i am on GPU mode or not, or refresh workspace, nothing is working

Profile photo
Profile photo
Profile photo
4 Replies Latest reply about 2 months ago

Mark R.

9:43 PM, Oct 22

Profile photo
@HimanshuM @GhanshyamY - I am struggling with loading the model in project 2. Since I am using a pre-trained network I am only passing in a new classifier to the vgg16 model. How do I load the pre-trained network when starting from the checkpoint? I can't find anything in previous lessons that describes how to do this. I have spent hours trying to find anything online that explains how to do this. Is there any direction you can provide me so I can continue to make progress on the project? Thanks

Profile photo
Profile photo
7 Replies Latest reply about 2 months ago

Michael R.

1:03 PM, Oct 23

Profile photo
is it normal for your logsoftmax to end up higher than one?

Profile photo
2 Replies Latest reply about 2 months ago

Michael R.

1:03 PM, Oct 23

Profile photo
test_loss = 0 images, labels = next(iter(dataloaders_test))

output = model.forward(images) test_loss += criterion(output, labels).item()

ps = torch.exp(output)

ps.shape ps.max(dim=1)

Michael R.

1:04 PM, Oct 23

Profile photo
the output

Michael R.

1:04 PM, Oct 23

Profile photo
(tensor(1.00000e-02 * [ 1.4656, 1.2382, 1.5270, 1.3735, 1.3872, 1.4223, 1.5958, 1.3177, 1.6511, 1.5047]), tensor([ 86, 9, 95, 58, 35, 87, 63, 0, 50, 89]))

Profile photo
1 Reply Latest reply about 2 months ago

Michael R.

1:04 PM, Oct 23

Profile photo
arent softmax probabilities supposed to be between 0 and 1?

arunabh S.

2:31 PM, Oct 23

Profile photo
Hi I am stuck on Torch.Compose on local system

arunabh S.

2:31 PM, Oct 23

Profile photo
Transform.Compose I mean

Profile photo
Profile photo
5 Replies Latest reply about 2 months ago

arunabh S.

2:31 PM, Oct 23

Profile photo
i have installed torch-cpu and torchvision

arunabh S.

2:31 PM, Oct 23

Profile photo
yet this problem exists

arunabh S.

2:31 PM, Oct 23

Profile photo
any solutions?

Thomas C.

8:23 PM, Oct 23

Profile photo
I am having trouble with backward passes though my model.

Thomas C.

8:23 PM, Oct 23

Profile photo
I am using Resnet 18, with an added classifer per our lectures

Thomas C.

8:24 PM, Oct 23

Profile photo
Forward pass works fine

Thomas C.

8:24 PM, Oct 23

Profile photo
during the backward pass I get the error element 0 of tensors does not require grad and does not have a grad_fn

Thomas C.

8:24 PM, Oct 23

Profile photo
before I loaded the classifer I froze all the layers and ufroze the classifier layers after adding them

Thomas C.

8:24 PM, Oct 23

Profile photo
i am not sure what I am doing wrong

Thomas C.

8:26 PM, Oct 23

Profile photo
Is there attribute I can look at the require_grad for each layer to see what the issue is?

Thomas C.

8:39 PM, Oct 23

Profile photo
NVM , true to form. 1) Struggle on something for days. 2) Ask for help 3) figure it out in 5 minutes while waiting for feedback.

Himanshu M.

Mentor

10:55 PM, Oct 23

Profile photo
@ThomasC Apologies, for delayed reply, but we mentors also have a few limitations regarding time, I am trying my best to answer all the questions as quickly as possible, as you can see there are just 2 mentors and there are over 198 students in this study group itself, and due to different timezones this happens. Meanwhile, I would suggest you to ask your queries by maintaining a single thread for a question, this helps me to followup on a thread quicker. If you are unaware you can use shift + enter to go to next line, this helps your questions to stay in one place.

Also, I would appreciate if you provide code snippet, while asking question, this helps me debug better.

I hope you follow this for better experience. Thank you.

Profile photo
3 Replies Latest reply about 2 months ago

Vidhu Shekhar S.

11:47 PM, Oct 23

Profile photo
I am getting below error while training the network in command line. Same code worked fine in Jupyter

cuda runtime error (2) : out of memory at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCStorage.cu:58

Profile photo
Profile photo
Profile photo
6 Replies Latest reply about 2 months ago

Vemula Dilip K.

5:12 AM, Oct 24

Profile photo
correct = 0 total = 0 with torch.no_grad(): for data in testloader: images, labels = data outputs = model.(images) _, predicted = torch.max(outputs.data, 1) total += labels.size(0) correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

Above given code is giving the below-given error, kindly suggest how can I resolve this issue Transfer Learning

RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'

Profile photo
Profile photo
5 Replies Latest reply about 2 months ago

Jennifer W.

11:10 AM, Oct 24

Profile photo
Hi everyone, I'm stuck on part 2 of project 2. I get that I need to use argparse and have written code for that. But when I type in 'python train.py <data_dir>' with or without optional arguments, nothing happens. It appears my script is not being run. Clearly I'm missing something.

Can someone point me in the right direction? Thanks

Profile photo
Profile photo
11 Replies Latest reply about 2 months ago

Michael R.

12:03 PM, Oct 24

Profile photo
For some reason when I train my data and run validation on it i get good results. However, when i save it and load it again , i get strange results and i cant seem to figure out why

Michael R.

12:03 PM, Oct 24

Profile photo
`

Michael R.

12:03 PM, Oct 24

Profile photo
` checkpoint = {'n_in': 25088, 'n_out': 102, 'n_h': [4096, 800], 'state_dict': model.state_dict(), 'epochs': 7, 'optimizer_state.dict': optimizer.state_dict(), 'class_to_idx:': image_datasets.class_to_idx}

saving the checkpoint configuration to disk
torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(filepath): checkpoint = torch.load(filepath) classifier = nn.Sequential(OrderedDict([ ('fc1', nn.Linear(n_in, n_h[0])),
('relu', nn.ReLU()), ('dropout', nn.Dropout(p=0.5)), ('fc2', nn.Linear(n_h[0],n_h[1])), ('relu2', nn.ReLU()), ('dropout2', nn.Dropout(p=0.5)), ('fc3', nn.Linear(n_h[1], n_out)), ('output', nn.LogSoftmax(dim=1)) ]))

model.classifier = classifier

model.checkpoint = {'n_in': 25088,
                     'n_out': 102,
                     'n_h': [4096, 800],
                     'state_dict': model.state_dict(),
                     'epochs': 7,
                     'optimizer_state.dict': optimizer.state_dict(),
                     'class_to_idx:': image_datasets.class_to_idx}

return model
model = load_checkpoint('checkpoint.pth')

def calculate_acc(model, data): model.eval() model.to('cuda')

with torch.no_grad():
    for idx, (inputs, labels) in enumerate(data):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        obtain outputs from the model
        outputs = model.forward(inputs)

        get the probabilities
        _, predicted = outputs.max(dim=1)

        if idx == 0:
            print(predicted) #the predicted class
            print(torch.exp(_)) # the predicted probability
        equals = predicted == labels.data

        if idx == 0:
            print(equals)
        print(equals.float().mean())
calculate_acc(model, dataloaders_valid) `

Profile photo
Profile photo
6 Replies Latest reply about 2 months ago

Rebecca B.

7:59 PM, Oct 24

Profile photo
Wondering if anyone has any feedback on some of the key points of discussion in https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw article including the suggestion that most problems don't need a second hidden layer (and even less more than that) and the equation that is provided to suggest the number of nodes for the hidden layer.

arunabh S.

8:37 AM, Oct 25

Profile photo
I am unable to install pytorch on windows

arunabh S.

8:37 AM, Oct 25

Profile photo
Also I am unable to use udacity remote notebooks for the function Transofrms.COmpose

Profile photo
Profile photo
Profile photo
22 Replies Latest reply about 2 months ago

arunabh S.

8:38 AM, Oct 25

Profile photo
can anyone help?

Vemula Dilip K.

9:43 PM, Oct 25

Profile photo
how to install and use fc_model in pytorch because of this I am unable save and load data? which was used in Saving and Loading Trained Networks topic

Profile photo
Profile photo
2 Replies Latest reply about 2 months ago

Samuel Alexander R.

10:58 PM, Oct 25

Profile photo
Why my vgg19 model is giving me a size size mismatch error? RuntimeError: size mismatch, m1: [40 x 25088], m2: [1024 x 500] at c:\programdata\miniconda3\conda-bld\pytorch_1533094653504\work\aten\src\thc\generic/THCTensorMathBlas.cu:249

# model selected
model = models.vgg19_bn(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(500,300)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(p=0.5)),
    ('fc3', nn.Linear(300, 121)),
    ('output', nn.LogSoftmax(dim=1))
]))

'''classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 500)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(500, 121)),
    ('output', nn.LogSoftmax(dim=1))
]))'''

model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

model.to(device)

import time

start = time.time()

for e in range(epochs):
    running_loss = 0
    for steps, (images, labels) in enumerate(dataloaders):

        # move images and labels to device selected
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                 "Loss: {:.4f}".format(running_loss/print_every))
            running_loss = 0
print('Finished Training in ', time.time() - start, 'seconds')
Profile photo
Profile photo
5 Replies Latest reply about 2 months ago

Vemula Dilip K.

4:49 AM, Oct 26

Profile photo
How to download Cat_Dog_data? I am Unable to download the data kindly suggest me.

Wei Chun C.

5:05 AM, Oct 26

Profile photo
Hi,

I use the following code to train my neural network but encounter a problem. Can anyone help me?

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=0.01)

sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model, criterion, optimizer, sched, 10, 'cuda')
Thanks!

Profile photo
Profile photo
12 Replies Latest reply about 2 months ago

Rebecca B.

11:59 AM, Oct 26

Profile photo
@HimanshuM I'm looking to run the project on my local machine and was all set up to install CUDA and cuDNN when I started tracking down more info about PyTorch and it would seem that I don't need to independently install CUDA?

If I run

torch.cuda.is_available()
I get True.

And

a = torch.full((10,), 3, device=torch.device("cuda"))
executes just fine.

Am I good to go for the project? What about in the long run?

Profile photo
1 Reply Latest reply about 2 months ago

Rebecca B.

3:02 PM, Oct 26

Profile photo
In the final section of the PyTorch section I am getting a RuntimeError that seems to be associated with getting the LogSoftmax function but I can't seem to work it out. The error says:

Given input size: (1024x3x3). Calculated output size: (1024x-3x-3). Output size is too small
Here is the code that I have used:

# Initial loading and transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(100),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()],
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225]))


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
# Import model
model = models.densenet121(pretrained=True)
# Create our classifier
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

# Replace the densenet classifier with our classifier
model.classifier = classifier
for device in ['cpu', 'cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break

    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")
I cannot work out why the negative values are occurring.

Profile photo
1 Reply Latest reply about 2 months ago

Mark R.

10:08 AM, Oct 27

Profile photo
@HimanshuM - what could explain why my predict function never correctly identifying the right flower name? https://cl.ly/53d0420c89cd

Profile photo
Profile photo
13 Replies Latest reply about 2 months ago

Samuel Alexander R.

9:31 PM, Oct 28

Profile photo
Reviewer gave me this feedback: You need to use PIL library and python native functions to complete this part so that you can learn more about one of the most used python libraries in image processing. Where I can find a tutorial to learn how to process a single image with the PIL and python native functions? I have managed to resize and crop the images. But how do I normalize and create a tensor?

Profile photo
Profile photo
21 Replies Latest reply about 2 months ago

Vemula Dilip K.

7:15 AM, Oct 29

Profile photo
for ii,(inputs, labels) in enumerate(trainloader):? how the given command is working when I am using in my local system it's throwing the error how data will move to multiple variables?

Profile photo
Profile photo
Profile photo
4 Replies Latest reply about 2 months ago

Thomas C.

11:57 AM, Oct 29

Profile photo
is there a way of searching conversation on this, so that I can see if a topic has already been covered?

Profile photo
Profile photo
4 Replies Latest reply about 2 months ago

Thomas C.

12:05 PM, Oct 29

Profile photo
I am trying to set up and train a resnet model.
In the previous class excersize we used densenet and replaced the classifier with a couple of fully connected layers with ReLU activations.
Resnet seems to not have a "classifier" but has a FC layer at the end.

question 1) I am guessing for the resnet we should be replacing this last layer instead of a classifier compared to the densenet model. Is this correct.

Question 2) What is the rule of thumb when adding classification layers? When I looked up training Resnet pre-trained models on both the Medium and Pytorch.org the tutorials were only showing modifying the last layer so the output meets the number of classes.

Profile photo
Profile photo
17 Replies Latest reply about 2 months ago

Ravish C.

12:24 PM, Oct 29

Profile photo
Opening my project and I get a message "Your Service is not running. Please double check to make sure your service is running in order to preview your project."

Ravish C.

12:25 PM, Oct 29

Profile photo
What is this error and how do I start the "service" again?

Profile photo
1 Reply Latest reply about 2 months ago

Rebecca B.

1:46 PM, Oct 29

Profile photo
I used vgg19_bn as the pretrained model to build my classifier. I've trained the model and saved the state_dict but when I try to load it I get the following error: AttributeError: 'VGG' object has no attribute 'load_state'

I've tried searching the error but wasn't able to find anyone who had the same issue.

Here is the code I used to save the model:

torch.save(model.state_dict(), 'checkpoint_adam.pth')
And here is the code I attempted to use to load it:

adam_state_dict = torch.load('checkpoint_adam.pth')
model.load_state_dict(adam_state_dict)
I tested the loading of the state_dict and I am able to see the keys, so it has definitely loaded, I just can't load it into the model.

Profile photo
Profile photo
2 Replies Latest reply about 2 months ago

Michael R.

2:05 PM, Oct 29

Profile photo
so I am really confused about something

Michael R.

2:05 PM, Oct 29

Profile photo
I have a checkpoint saved in my folder that i originally trained but i cannot use it on the command line

Profile photo
1 Reply Latest reply about 2 months ago

Michael R.

2:05 PM, Oct 29

Profile photo
is this not supposed to work?

Michael R.

7:13 PM, Oct 29

Profile photo
getting 'this service is not running' error message page

Profile photo
1 Reply Latest reply about 2 months ago

ibrahim G.

10:43 PM, Oct 29

Profile photo
same thing happening to me at the moment

ibrahim G.

10:44 PM, Oct 29

Profile photo
@HimanshuM - is there a github repo containing all the data files?

Profile photo
1 Reply Latest reply about 2 months ago

ibrahim G.

10:44 PM, Oct 29

Profile photo
https://github.com/udacity/aipnd-project has an assets folder which doesnt contain any image data.

Balachandar P.

12:31 AM, Oct 30

Profile photo
@HimanshuM ,

Balachandar P.

12:32 AM, Oct 30

Profile photo
@HimanshuM , vgg16=models.vgg16(pretrained=True) vgg16 for param in vgg16.parameters(): param.requires_grad=False

classifier= nn.Sequential(OrderedDict([
('first',nn.Linear(25088,4096)),
('first_relu',nn.ReLU()),
('first_drop',nn.Dropout(0.1)),
('second',nn.Linear(4096,2000)),
('second_relu',nn.ReLU()),
('second_drop',nn.Dropout(0.2)),
('third',nn.Linear(2000,1000)),
('third_relu',nn.ReLU()),
('third_drop',nn.Dropout(0.3)),
('fourth',nn.Linear(1000,500)),
('fourth_relu',nn.ReLU()),
('fourth_drop',nn.Dropout(0.2)),
('fifth',nn.Linear(500,102)),
('fifth_log',nn.LogSoftmax(dim=1)),
    ]))
vgg16.classifier=classifier
vgg16
vgg16.to('cuda')
loss=nn.NLLLoss()
optimizer=optim.Adam(vgg16.classifier.parameters(),lr=0.01)
steps=0
running_loss=0
for e in range(20):
    for image,label in iter(dataloaders_train):
        steps += 1
        image,label=image.to('cuda'),label.to('cuda')
        optimizer.zero_grad()
        output=vgg16.forward(image)
        po=loss(output,label)
        po.backward()
        optimizer.step()
        running_loss += po.data[0]
        if steps % 40 == 0:
            print("Epoch :{}".format(e+1))
            print("Training Loss: {:.3f}.. ".format(running_loss/40))
            running_loss=0
Balachandar P.

12:33 AM, Oct 30

Profile photo
The above code is the code, I am using. And even if change any of the parameters like epoch,learning rate or dropout probability. I end up with loss of 4.

Balachandar P.

12:33 AM, Oct 30

Profile photo
4.490.. not less than that.. please help me out.. I am trying very hard to reduce the loss in all ways.. but nothing is working out..

Balachandar P.

12:34 AM, Oct 30

Profile photo
Please help me as early as possible. I am reaching out here, after trying all the possibilities..

Profile photo
Profile photo
3 Replies Latest reply about 2 months ago

Balachandar P.

12:36 AM, Oct 30

Profile photo
And the transforms I used are,

Balachandar P.

12:36 AM, Oct 30

Profile photo
data_train_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.RandomGrayscale(p=0.2),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]) data_test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

TODO: Load the datasets with ImageFolder
image_train_datasets = datasets.ImageFolder(train_dir,transform=data_train_transforms) image_validation_datasets=datasets.ImageFolder(valid_dir,transform=data_test_transforms) image_test_datasets=datasets.ImageFolder(test_dir,transform=data_test_transforms)

TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders_train = torch.utils.data.DataLoader(image_train_datasets,batch_size=32,shuffle=True) dataloaders_valid = torch.utils.data.DataLoader(image_validation_datasets,batch_size=32,shuffle=True) dataloaders_test = torch.utils.data.DataLoader(image_test_datasets,batch_size=32,shuffle=True)

Balachandar P.

8:29 AM, Oct 30

Profile photo
@Himanshu please help me atleast with predict function

Balachandar P.

8:41 AM, Oct 30

Profile photo
After normalisation, MY RGB values are not between 0 and 1. image="flowers/train/1/image_06734.jpg" im = Image.open(image) resized_image=im.resize((256,256)) width, height = resized_image.size # Get dimensions new_width,new_height=224,224 left = (width - new_width)/2 top = (height - new_height)/2 right = (width + new_width)/2 bottom = (height + new_height)/2 cropped_image=resized_image.crop((left, top, right, bottom)) np_image=np.asarray(cropped_image) mean = np.array([0.485, 0.456, 0.406]) std = np.array([0.229, 0.224, 0.225]) final_np=(np_image-mean)/std c=final_np.transpose((2,0,1))

Balachandar P.

8:41 AM, Oct 30

Profile photo
@HimanshuM , Please reply for the above query atleast

Profile photo
2 Replies Latest reply about 2 months ago

Matthew M.

11:41 AM, Oct 30

Profile photo
Anyone having trouble connecting to the Juypter notebook server? Been trying for about a week and contacted Udacity support. Crickets...

Profile photo
Profile photo
2 Replies Latest reply about 2 months ago

Balachandar P.

8:38 AM, Oct 31

Profile photo
@HimanshuM yes I asked a query.. Since there was delay in response, I tried myself and got the answer.. I do understand that 2 mentors are allocated for around 200 people. But since we have time constraint to complete the project, kindly help us.

Profile photo
1 Reply Latest reply about 2 months ago

Balachandar P.

8:40 AM, Oct 31

Profile photo
@HimanshuM , I am right now in the step of applying an image for the built model. Each time the 5 high probability classes are changing(not very often but changing). image_path='flowers/test/11/image_03098.jpg' processed_image=process_image(image_path) #3 224 224 print(processed_image.shape) trans_apply=transforms.Compose([transforms.ToTensor()]) tens=trans_apply(processed_image) tens=tens.permute(1,0,2)

tens=Variable(tens.unsqueeze(0))

output=model.forward(tens.float())image_path='flowers/test/11/image_03098.jpg' processed_image=process_image(image_path) #3 224 224 print(processed_image.shape) trans_apply=transforms.Compose([transforms.ToTensor()]) tens=trans_apply(processed_image) tens=tens.permute(1,0,2)

tens=Variable(tens.unsqueeze(0))

output=model.forward(tens.float())

Am I doing anything wrong here? Please help.

Profile photo
1 Reply Latest reply about 2 months ago

Rebecca B.

9:49 AM, Oct 31

Profile photo
@HimanshuM I've been able to successfully convert my image array from HWC to CHW and it now has the dimensions 3 x 224 x 224. However, when I try to feed the tensor into the model, it doesn't work because the model input dimensions are 64 x 3 x 3 x 3. I have no idea how to fix this.

Profile photo
Profile photo
Profile photo
6 Replies Latest reply 22 days ago

arunabh S.

10:24 AM, Oct 31

Profile photo
@HimanshuM I am unable to resolve:

arunabh S.

10:24 AM, Oct 31

Profile photo
cuda runtime error (59) : device-side assert triggered at /opt/conda/conda-bld/pytorch_1524584710464/work/aten/src/THC/generic/THCStorage.c:36

Profile photo
Profile photo
13 Replies Latest reply about 2 months ago

Rebecca B.

1:40 PM, Oct 31

Profile photo
I've been able to successfully able to plot the image and the top 5 predictions but I'd love to be able to get the plot areas to align. (https://www.dropbox.com/s/9hrhhlmy7iqsr7q/example_plot.png?dl=0) Anyone know how to do that successfully?

Wei Chun C.

7:39 PM, Oct 31

Profile photo
Hi, I'm confused about how to start project part 2. Can anyone help me out? My understanding is that in train.py, I need to load the model I save in part1. And in predict.py, I need to design to make it for user to use the model in train.py. Am I right? And also, how should I try run the code in the part2 workplace?

Profile photo
Profile photo
3 Replies Latest reply about 2 months ago

arunabh S.

5:35 AM, Nov 1

Profile photo
@HimanshuM that worked however got stuck at the following now. i am googling and trying to understand my code but could not find a solution to the following error:

arunabh S.

5:35 AM, Nov 1

Profile photo
RuntimeError: Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'

Profile photo
Profile photo
14 Replies Latest reply about 2 months ago

arunabh S.

5:35 AM, Nov 1

Profile photo
I am attaching link to my new code. please have a look and let me know whats wrong:

Profile photo
1 Reply Latest reply about 2 months ago

arunabh S.

5:36 AM, Nov 1

Profile photo
https://github.com/arunabh15091989/dsnd-pytorch-issue/blob/master/Image%20Classifier%20Project.md

arunabh S.

5:36 AM, Nov 1

Profile photo
also the notebook is here: https://github.com/arunabh15091989/dsnd-pytorch-issue/blob/master/Image%20Classifier%20Project.ipynb

Himanshu M.

Mentor

1:50 PM, Nov 1

Profile photo
Hi Everyone, I will be travelling for next day or two and possibly I won't be able to reply to you quickly. Though, I will try to help whenever I get time. Kindly have patience. Thank you.

Jing M.

3:00 AM, Nov 2

Profile photo
how to load a single image in torch given an image_path?

Profile photo
1 Reply Latest reply about 2 months ago

Maria F.

6:43 AM, Nov 2

Profile photo
Hey everyone in chapter Neural Networks in Keras,for XOR problem how did we get 32 nodes in solution when it is mentioned "Set the first layer to a Dense() layer with an output width of 8 nodes and the input_dim set to the size of the training samples (in this case 2)." and output as 2 when "Set the output layer width to 1, since the output has only two classes. (We can use 0 for one class and 1 for the other)"... am i missing something?

arunabh S.

7:24 AM, Nov 2

Profile photo
I am getting the error : your service is not running. @GhanshyamY #udacity pls help

Profile photo
Profile photo
Profile photo
3 Replies Latest reply about 2 months ago

Rebecca B.

2:39 PM, Nov 2

Profile photo
For Part 2 of the project I'd never encountered argparse before and found the tutorial linked in the class hard to follow without some foundational background. I discovered there's a more introductory tutorial in the Python documentation and can be found here: https://docs.python.org/3/howto/argparse.html I found it much easier to start here and then go back to the other tutorial!

Profile photo
1 Reply Latest reply about 2 months ago

Mark R.

4:03 PM, Nov 3

Profile photo
I am struggling with the argparse... How do I get those arguments to update the values in my code? When I am creating an argument am I declaring a variable?

Profile photo
Profile photo
Profile photo
4 Replies Latest reply about 1 month ago

Wei Chun C.

11:07 PM, Nov 3

Profile photo
Hi there, I got stuck in part 2 of the project. I wonder if you can help me out.

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict

parser = argparse.ArgumentParser(description='train.py parser')

parser.add_argument('data_dir', type=str,help='Data Directory')
parser.add_argument('--save_dir', type=str,help='Save data to directory')
parser.add_argument('--arch', type=str,help='Architecture for the model')
parser.add_argument('--learning_rate', type=float,help='Set learning rate')
parser.add_argument('--hidden_units', type=int,help='Set hidden units')
parser.add_argument('-epochs', type=int,help='Set epochs')
parser.add_argument('--gpu', action='store_true',help='Use GPU if available')

args = parser.parse_args()

model_old = torch.load('checkpoint.pth')
model = models.vgg16(pretrained=True)
model.classifier = model_old['classifier']
model.class_to_idx = model_old['class_to_idx']
model.load_state_dict(model_old['state_dict'])

return model
I'm not sure if I should load the model I have in part 1 or try to build a new one with the --arch. I know either way I need to train it and print out the loss, validation loss, and validation accuracy. Hope you can give me some suggestions.

Profile photo
Profile photo
3 Replies Latest reply about 1 month ago

Mark R.

12:55 PM, Nov 4

Profile photo
I am working on the predict.py file but I keep getting this runtime error when running that file: RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same. I never experienced this error in my jupyter notebook. Could you please explain why this may be occurring @HimanshuM ? Thanks

Profile photo
Profile photo
Profile photo
7 Replies Latest reply about 1 month ago

Maria F.

11:15 PM, Nov 4

Profile photo
In chapter Analyzing IMDB Data in Keras what is tokenizer doing?When we are performing one hot encoding how is the matrix filled. eg we have vector [1, 11, 2, 11, 4, 2, 745, 2, 299, 2, 590, 2, 2, 37]. How will it be converted to matrix of 1's and 0's.How are repeated indexes are treated like 2 is repeated here in the vector @GhanshyamY @HimanshuM

Profile photo
Profile photo
3 Replies Latest reply about 1 month ago

Ravi Kishore N.

1:40 PM, Nov 5

Profile photo
HI @HimanshuM @GhanshyamY ima unable to know significanse of _, preds = torch.max(outputs.data, 1) .why the naming convention is like this

Profile photo
1 Reply Latest reply about 1 month ago

Jennifer W.

2:35 PM, Nov 5

Profile photo
Hi everyone, hope things are going well. I am really stuck on Part 2. No matter what I type in the command line prompt in the format 'python train.py' + [whatever], nothing happens, the command line prompt pops up immediately again.

Does anyone know what's happening? I feel I must be missing something obvious, but I've been stuck here a while.

Profile photo
Profile photo
Profile photo
Profile photo
15 Replies Latest reply 25 days ago

Thomas C.

4:11 PM, Nov 5

Profile photo
I am having trouble training my model, now that I have put my model training algorithm into a callable function. It converges first time I run the function but, subsequent times it doesn't seems to be converging toward anything. no matter how many epochs I run, or how many hyperparameters I change, or what data augmentation (e.g. rotations. My training loss, test_loss, and accuracy doesnt really change.

Does anyone have any advice on what I should check?

Profile photo
Profile photo
Profile photo
29 Replies Latest reply about 1 month ago

Karthik V.

5:14 AM, Nov 6

Profile photo
In the image classifer project, why are there both test and validation datasets. Is the idea that we use the validation set for hyperparameter tuning ( number of layers in the model, dropout etc ) and finally compute the performance on the test dataset?

Seemingly the same question was asked on knowledge ( https://knowledge.udacity.com/questions/10799 ). I couldnt quite understand the answer there though

Knowledge Post

Test vs Validation
{__('knowledge answers')}
1 ANSWER
Profile photo
2 Replies Latest reply about 1 month ago

arunabh S.

11:54 AM, Nov 6

Profile photo
process image

arunabh S.

12:03 PM, Nov 6

Profile photo
Can someone explain what is to be done here:And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using ndarray.transpose. The color channel needs to be first and retain the order of the other two dimensions

Profile photo
1 Reply Latest reply about 1 month ago

arunabh S.

12:04 PM, Nov 6

Profile photo
@HimanshuM can u help with above and also how do i write the predict function. I am not sure about it. it is pretty confusing the directions are not clear on completing the function

Profile photo
2 Replies Latest reply about 1 month ago

Anindya M.

12:08 AM, Nov 8

Profile photo
I am confused at gradient descent concepts in deep learning section. The instructor considered a MSE error function and calculated the gradient descent steps, but in the first section of the course the probabilistic error function was chosen to calculate the gradient descent step. So, for each problem, will there be different gradient descent steps?

Profile photo
Profile photo
3 Replies Latest reply about 1 month ago

Ravi Kishore N.

9:45 AM, Nov 10

Profile photo
Hi @HimanshuM @GhanshyamY iam getting following error in last question of project1 image classifier. Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight. Can please let me know issue here as saved model is working fine on test set .

Profile photo
Profile photo
8 Replies Latest reply 29 days ago

samarth S.

5:02 PM, Nov 10

Profile photo
If you import torch then do you really need to import nn and optim again from torch?

Profile photo
Profile photo
2 Replies Latest reply about 1 month ago

arunabh S.

4:49 AM, Nov 11

Profile photo
@HimanshuM what is the difference between 'cuda:0', 'cuda' and device and when to use which. please let me know I am confused about it

Profile photo
2 Replies Latest reply about 1 month ago

arunabh S.

4:49 AM, Nov 11

Profile photo
right now i just google and if it says cuda:0 i use that

arunabh S.

4:49 AM, Nov 11

Profile photo
i have no clear understanding of it

Maria F.

5:10 AM, Nov 11

Profile photo
@GhanshyamY @HimanshuM I am getting very low accuracy with validation set. this is the classifier I have chosen classifier=nn.Sequential(OrderedDict([ ('fc1',nn.Linear(in_features=25088,out_features=4096)), ('relu1',nn.ReLU()), ('drop1',nn.Dropout(p=0.5)), ('fc2',nn.Linear(in_features=4096,out_features=1000)), ('relu2',nn.ReLU()), ('drop2',nn.Dropout(p=0.5)), ('fc3',nn.Linear(in_features=1000,out_features=102)), ('output',nn.LogSoftmax(dim=1)) ])) and this is optimizer and criterion: criterion=nn.NLLLoss() optimizer=optim.Adam(model.classifier.parameters(),lr=0.01) Am i doing something wrong???

Profile photo
Profile photo
Profile photo
12 Replies Latest reply about 1 month ago

Karthik V.

5:25 AM, Nov 11

Profile photo
I am trying to solve the main image classifier project. I am having difficulty getting my model to have a train accuracy of even 3%. I went through previous questions on this thread. I tried the following:

Using pretrained models for vgg16 and densenet121
Using a learning rate of 1e-1 to 1e-6
Using number of hidden layers between 2 and 4 ( I looked at this thread that suggested keeping the model simple https://study-hall.udacity.com/sg-474340-2087/rooms/community:nd025:474340-cohort-2087-project-1663/community:thread-11635580657-261987?contextType=room )
I double checked the following:

model.eval() and model.train() around computing error and accuracy on validation set
Reset training_loss and training_accuracy to 0 after printing them
Set optimizer gradients to zero in each loop since we perform the optimizer step in each loop
Is there anything obvious I am missing out or I should just wait for many more epochs before expecting better results? In the MNIST, Fashion-MNIST and Cat/Dog examples our models converged pretty rapidly ( about 5 was plenty to reach 70%+ accuracy ). Would appreciate some help here

My code is present on github: https://github.com/karthikvijayakumar/Udacity_DSND_1/blob/master/Image%20Classifier%20Project.ipynb

Profile photo
Profile photo
Profile photo
13 Replies Latest reply about 1 month ago

arunabh S.

8:09 AM, Nov 13

Profile photo
@HimanshuM I am getting the following error in my predict.py:

arunabh S.

8:09 AM, Nov 13

Profile photo
|████████████████████████████████████████████████████████████████████████| 553433881/553433881 [00:13<00:00, 39945119.63it/s] Traceback (most recent call last): File "/opt/conda/lib/python3.6/site-packages/PIL/Image.py", line 2481, in open fp.seek(0) AttributeError: 'list' object has no attribute 'seek'

During handling of the above exception, another exception occurred:

Traceback (most recent call last): File "predict.py", line 95, in <module> probs, classes, top_flowers = predict(input_img, model, num_classes) File "predict.py", line 55, in predict img = process_image(image_path) File "predict.py", line 26, in process_image img = Image.open(image_path) File "/opt/conda/lib/python3.6/site-packages/PIL/Image.py", line 2483, in open fp = io.BytesIO(fp.read()) AttributeError: 'list' object has no attribute 'read'

arunabh S.

8:10 AM, Nov 13

Profile photo
python predict.py --gpu True input_img '/home/workspace/aipnd-project/flowers/test/13/image_05787.jpg' --top_k 4

arunabh S.

8:10 AM, Nov 13

Profile photo
I used the above command and arguments for predict.py

arunabh S.

8:41 AM, Nov 13

Profile photo
@HimanshuM @GhanshyamY please look at my code below: https://github.com/arunabh15091989/image-classifier-issues

arunabh S.

8:42 AM, Nov 13

Profile photo
and let me know what the above issue is happening. I cant seem to figure it out so I am moving ahead with unsupervised learning section for now

Profile photo
Profile photo
9 Replies Latest reply about 1 month ago

Maria F.

5:08 AM, Nov 14

Profile photo
@HimanshuM Hi mentor when I try perform imports in workspace i get the below Traceback (most recent call last): File "train.py", line 2, in <module> from utily import data_load,nn_model ModuleNotFoundError: No module named 'utily'

I have tried doing import sys and then sys.path.append(directory) but still getting error, both utily and train are in the same directory

Profile photo
Profile photo
Profile photo
5 Replies Latest reply about 1 month ago

Yichun T.

9:01 PM, Nov 14

Profile photo
@HimanshuM Hi, mentor. I was trying to do the Image Preprocessing part of the project, but encountered some error when I tried to test out the two functions, def process_image(image) and def imshow(image, ax=None, title=None). The following is my definition of the two functions. def process_image(image): ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array '''

# TODO: Process a PIL image for use in a PyTorch model
transform = transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
pil_image = datasets.ImageFolder(image , transform=transform)
np_image = np.array(pil_image)
np_image.transpose

return np_image
Yichun T.

9:03 PM, Nov 14

Profile photo
def process_image(image): ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array '''

# TODO: Process a PIL image for use in a PyTorch model
transform = transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
pil_image = datasets.ImageFolder(image , transform=transform)
np_image = np.array(pil_image)
np_image.transpose

return np_image
Yichun T.

9:03 PM, Nov 14

Profile photo
def imshow(image, ax=None, title=None): if ax is None: fig, ax = plt.subplots()

# PyTorch tensors assume the color channel is the first dimension
# but matplotlib assumes is the third dimension
image = image.transpose((1, 2, 0))

# Undo preprocessing
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image = std * image + mean

# Image needs to be clipped between 0 and 1 or it looks like noise when displayed
image = np.clip(image, 0, 1)

ax.imshow(image)

return ax
Yichun T.

9:04 PM, Nov 14

Profile photo
imshow(process_image('/home/workspace/aipnd-project/assets/Preprocessing/'))

Yichun T.

9:05 PM, Nov 14

Profile photo
RuntimeError Traceback (most recent call last) <ipython-input-26-800d2bdfef37> in <module>() ----> 1 imshow(process_image('/home/workspace/aipnd-project/assets/Preprocessing/'))

<ipython-input-16-fa02b853cf74> in process_image(image) 11 transforms.Normalize([0.485, 0.456, 0.406], 12 [0.229, 0.224, 0.225])]) ---> 13 pil_image = datasets.ImageFolder(image , transform=transform) 14 np_image = np.array(pil_image) 15 np_image.transpose

/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/datasets/folder.py in init(self, root, transform, target_transform, loader) 176 super(ImageFolder, self).init(root, loader, IMG_EXTENSIONS, 177 transform=transform, --> 178 target_transform=target_transform) 179 self.imgs = self.samples

/opt/conda/lib/python3.6/site-packages/torchvision-0.2.1-py3.6.egg/torchvision/datasets/folder.py in init(self, root, loader, extensions, transform, target_transform) 77 if len(samples) == 0: 78 raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n" ---> 79 "Supported extensions are: " + ",".join(extensions))) 80 81 self.root = root

RuntimeError: Found 0 files in subfolders of: /home/workspace/aipnd-project/assets/Preprocessing/ Supported extensions are: .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif

Profile photo
Profile photo
Profile photo
Profile photo
15 Replies Latest reply about 1 month ago

Gareth E.

8:14 AM, Nov 15

Profile photo
Hi, I've looked through my code thoroughly but cannot figure out what is wrong. My test accuracy remains at around 1 - 3% and does not noticeably improve between epocs. I'm pretty sure the architecture of the model is correct but I have some other bug in my code. Any help would be much appreciated. Here is the code:

Profile photo
Profile photo
Profile photo
Profile photo
52 Replies Latest reply 25 days ago

arunabh S.

7:00 AM, Nov 16

Profile photo
@GhanshyamY facing new issue after removing nargrs:

arunabh S.

7:00 AM, Nov 16

Profile photo
python predict.py input_img /home/workspace/aipnd-project/flowers/test/15/image_06374.jpg --gpu True --top_k 4 Traceback (most recent call last): File "predict.py", line 93, in <module> model,arch,num_classes = utility.load_checkpoint(path) File "/home/workspace/paind-project/utility.py", line 165, in load_checkpoint checkpoint = torch.load(filepath) File "/opt/conda/lib/python3.6/site-packages/torch/serialization.py", line 303, in load return _load(f, map_location, pickle_module) File "/opt/conda/lib/python3.6/site-packages/torch/serialization.py", line 459, in _load magic_number = pickle_module.load(f) _pickle.UnpicklingError: invalid load key, '\xff'.

arunabh S.

9:41 AM, Nov 17

Profile photo
Unable to review Your project could not be reviewed. Please resubmit after you address the issue noted below by the reviewer.

Dear Student,

The notebook you have submitted has some errors. It is probably because you ran the cells in different order. Please rerun your entire notebook again and resolve these errors then submit again.

Note:

When you are training the network, you also have to print Validation loss and accuracy along with training loss for each epoch. Also, you have to sum up the testing accuracy and print final accuracy. example 70.02%. Please take your SG mentor's help if required. Good Luck in your next submission

arunabh S.

9:41 AM, Nov 17

Profile photo
@HimanshuM @GhanshyamY I get the above message when i submit. not sure what is expected of me to submit project correctly.

Profile photo
Profile photo
5 Replies Latest reply 29 days ago

GAURAV A.

1:12 PM, Nov 19

Profile photo
In Transfer Learning solution lab(i.e. 11. Transfer Learning solution), what is the "Accuracy of the network on the 10000 test images". I am getting accuracy of 51%. Also i can see the total images to be 2500 rather than 10000. Did anyone faced the similar issue?

Profile photo
Profile photo
2 Replies Latest reply 29 days ago

Liza D.

12:54 AM, Nov 21

Profile photo
hi there, i am having trouble transferring my notebook and html file from workspace 1 to workspace 2 in project. when i go to next from workspace 1 it is not taking all my new files and checkpoints with. anyone else experience that?

Gareth E.

11:11 AM, Nov 23

Profile photo
Has anybody had issue trying to Save the checkpoint? I am getting a "No space left on device error" when running on Udacity's server

Profile photo
Profile photo
13 Replies Latest reply 24 days ago

Aastha ..

11:43 AM, Nov 23

Profile photo
hey..i am getting a trouble in doing the part 2 of image classification project.I am really cnfused how to use the console and run the code written in both the files.Some help would be really appreciated

Profile photo
1 Reply Latest reply 25 days ago

Maria F.

10:09 AM, Nov 25

Profile photo
How do I submit directly from workspace? When I se

Profile photo
Profile photo
4 Replies Latest reply 21 days ago

Maria F.

10:10 AM, Nov 25

Profile photo
When I select submit, it asks me to upload a zip file or git hub

Dai C.

11:58 PM, Nov 25

Profile photo
Hi mentors, how does transpose in the image processing part work? my code as follows: def process_image(image): ''' Scales, crops, and normalizes a PIL image for a PyTorch model, returns an Numpy array '''

# TODO: Process a PIL image for use in a PyTorch model
img_loader = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
pil_image = Image.open(image)
pil_image = img_loader(pil_image)
np_image = np.array(pil_image)
np_image = np_image.transpose((2, 0, 1))
return np_image
Dai C.

12:00 AM, Nov 26

Profile photo
it gave me operands could not be broadcast together with shapes (3,) (3,224,224) error when I ran the imshow function

Profile photo
Profile photo
2 Replies Latest reply 21 days ago

Robert B.

12:10 AM, Nov 26

Profile photo
@HimanshuM Hi, Himanshu. I was trying to do the Image Preprocessing part of the project, but encountered a problem when I used the process_image function and then let the code display the image through imshow. The image is almost white but I can see some lines that show the shape of the flower. The code is ``` def process_image(image):

# Resize the images where shortest side is 256 pixels, keeping aspect ratio.
if image.width > image.height:
    factor = image.width/image.height
    image = image.resize(size=(int(round(factor*256,0)),256))
else:
    factor = image.height/image.width
    image = image.resize(size=(256, int(round(factor*256,0))))
# Crop out the center 224x224 portion of the image.
image = image.crop(box=((image.width/2)-112, (image.height/2)-112, (image.width/2)+112, (image.height/2)+112))
# Convert to numpy array
np_image = np.array(image)
# Normalize image
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
np_image = (np_image - mean) / std
# Reorder dimension for PyTorch
np_image = np.transpose(np_image, (2, 0, 1))
return np_image ```
Himanshu M.

Mentor

8:52 AM, Nov 26

Profile photo
after this :

Himanshu M.

Mentor

8:52 AM, Nov 26

Profile photo
np_image = np.array(image)

do this
np_image = np_image/255.

Himanshu M.

Mentor

8:53 AM, Nov 26

Profile photo
the mean and std values are for the image with values in range 0-1

Robert B.

9:02 AM, Nov 26

Profile photo
That's because of this, right? "Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1." Anyways, now it works, thanks a lot!

Profile photo
1 Reply Latest reply 22 days ago

Dai C.

12:56 PM, Nov 26

Profile photo
@HimanshuM and @GhanshyamY I don't understand why we need to transform the image manually; Isn't it the same as the transform we did in the first part? transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) appreciate your help!!

Profile photo
Profile photo
4 Replies Latest reply 22 days ago

neo N.

7:50 PM, Nov 26

Profile photo
hi all, what is the difference between nn.Relu() and nn.Relu.(inplace)?

Profile photo
Profile photo
4 Replies Latest reply 21 days ago

Dai C.

8:44 PM, Nov 26

Profile photo
@HimanshuM @GhanshyamY Hi mentors! I have a question in the model prediction. Here is my code: ``` image = process_image('flowers/test/2/image_05100.jpg') tensor_image = torch.from_numpy(image).type(torch.FloatTensor) model_input = tensor_image.unsqueeze(0)

model = load_checkpoint('checkpoint.pth')[0] model.eval() model.to('cpu') # why cuda does not work? probs = torch.exp(model.forward(model_input)) top_probs, top_labs = probs.topk(k=5) print(top_probs) print(top_labs) ``` My question is, when I used model.to('cuda') it gave me runtime error. Why do I need to change it to cpu?

Profile photo
1 Reply Latest reply 22 days ago

Florent C.

1:55 PM, Nov 28

Profile photo
Hello, For the image classifier, I am only getting a test accuracy of 2% and I am supposed to get 70%. Please see my code below: @HimanshuM @GhanshyamY

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(2048, 1024),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.05)

model.to(device);

epochs = 1
steps = 0
running_loss = 0
print_every = 50
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
Profile photo
Profile photo
Profile photo
9 Replies Latest reply 19 days ago

Dai C.

12:35 PM, Nov 29

Profile photo
How do I get started on Part2...

Dai C.

12:56 PM, Nov 29

Profile photo
@GhanshyamY @HimanshuM So I am trying to run argparse in terminal: parser = argparse.ArgumentParser(description='Training Image Classifier.') parser.add_argument('data_dir', help='image for training', default="data/flowers/") parser.add_argument('--save_dir', help='Model checkpoint to use for inference') args = parser.parse_args() and it returns me an error: : error: the following arguments are required: data_dir

Profile photo
Profile photo
2 Replies Latest reply 19 days ago

Florent C.

9:34 PM, Nov 29

Profile photo
I get the below error when trying to display a Pytorch tensor as an image using the imshow function:
TypeError: transpose(): argument 'dim0' (position 1) must be int, not tuple

inputs, labels = next(iter(testloader))
imshow(inputs[0])
inputs[0].shape:
torch.Size([3, 224, 224])
@GhanshyamY @HimanshuM

Profile photo
Profile photo
3 Replies Latest reply 18 days ago

Prakul A.

3:18 AM, Nov 30

Profile photo
Hi mentors,

Prakul A.

3:19 AM, Nov 30

Profile photo
I am getting 51% accuracy on transfer learning project even after using the same code as solution. Is this the expected outcome ?

Profile photo
Profile photo
Profile photo
5 Replies Latest reply 12 days ago

Vishi C.

9:42 AM, Nov 30

Profile photo
@HimanshuM @GhanshyamY For Part 2 of the Image Classifier - how do I create the new files (train.py and predict.py) in the workspace.

Profile photo
1 Reply Latest reply 18 days ago

Rebecca B.

1:38 PM, Nov 30

Profile photo
@HimanshuM @GhanshyamY There's a person in the Data Science AMA that doesn't seem to have access to the project channels and has an inquiry about how to save the files in the workspace. Would someone be able to help him?

Profile photo
1 Reply Latest reply 18 days ago

Florent C.

10:15 PM, Nov 30

Profile photo
@HimanshuM @GhanshyamY validation accuracy is 78.8% but test accuracy is 1%. Here is the code for the test accuracy:

test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)

        test_loss += batch_loss.item()

        # Calculate accuracy
        ps = torch.exp(logps)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

print(f"Test loss: {test_loss/len(testloader):.3f}.. "
      f"Test accuracy: {accuracy/len(testloader):.3f}")
here is the network architecture:

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(25088, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.1),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0003)

model.to(device);
Profile photo
Profile photo
2 Replies Latest reply 17 days ago

Florent C.

11:15 PM, Nov 30

Profile photo
@GhanshyamY @HimanshuM I am getting the below error when trying to do the forward pass on the processed image:
RuntimeError: expected stride to be a single integer value or a list of 1 values to match the convolution dimensions, but got stride=[1, 1]

code that generates the error:

image = Image.open(img_path)
input_npy = process_image(image)
input_tensor = torch.from_numpy(input_npy)
inputs = input_tensor.to(device)
with torch.no_grad():
        logps = model.forward(inputs)
The process_image() function works when testing with imshow()

Profile photo
Profile photo
6 Replies Latest reply 17 days ago

Dai C.

4:13 AM, Dec 1

Profile photo
@HimanshuM I have a super basic question about part 2... Do I need a function in the train.py? or if I just copy over part 1 & add in argparse

Profile photo
1 Reply Latest reply 17 days ago

Dai C.

4:16 AM, Dec 1

Profile photo
@HimanshuM also why do we have two directories? aipnd-project and paind-project

Profile photo
Profile photo
Profile photo
5 Replies Latest reply 14 days ago

Dai C.

6:27 AM, Dec 1

Profile photo
@GhanshyamY @HimanshuM Help! In part 2, when I am running training.py, I got the following error: Accuracy of the network on the test images: 81 % Traceback (most recent call last): File "paind-project/train.py", line 196, in <module> train_network(arch, epochs, gpu) File "paind-project/train.py", line 192, in train_network torch.save(checkpoint, save_dir) File "/opt/conda/lib/python3.6/site-packages/torch/serialization.py", line 161, in save return _with_file_like(f, "wb", lambda f: _save(obj, f, pickle_module, pickle_protocol)) File "/opt/conda/lib/python3.6/site-packages/torch/serialization.py", line 118, in _with_file_like return body(f) File "/opt/conda/lib/python3.6/site-packages/torch/serialization.py", line 161, in <lambda> return _with_file_like(f, "wb", lambda f: _save(obj, f, pickle_module, pickle_protocol)) File "/opt/conda/lib/python3.6/site-packages/torch/serialization.py", line 227, in _save pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol) TypeError: file must have a 'write' attribute

Profile photo
Profile photo
4 Replies Latest reply 17 days ago

Dai C.

5:02 PM, Dec 1

Profile photo
@GhanshyamY @HimanshuM Help! when running the predict.py, I am seeing the following error: root@dae512d8fe63:/home/workspace# python paind-project/predict.py aipnd-project/flowers/test/1/image_06743.jpg aipnd-project/checkpoint.pth usage: predict.py [-h] [--cat_to_name CAT_TO_NAME] [--image IMAGE] [--save_dir SAVE_DIR] [--topk TOPK] predict.py: error: unrecognized arguments: aipnd-project/flowers/test/1/image_06743.jpg aipnd-project/checkpoint.pth

Profile photo
1 Reply Latest reply 17 days ago

Melanie R.

8:05 AM, Dec 6

Profile photo
I'm working on the last step of part 1: I want to execute the following line of code: probs, classes = predict(image_path, model, topk=5) But I receive the following error Expected object of type torch.FloatTensor but found type torch.cuda.FloatTensor for argument #2 'weight'

model.cuda() or model.to('cuda') did not change anything

Profile photo
3 Replies Latest reply 12 days ago

Aleksandra D.

12:56 PM, Dec 6

Profile photo
Hi Mentors and Students! I have a problem with my NN. I"m using AlexNet but my validation accuracy is very low in comparison to the training and test results. I'm not sure what could be the problem there. Do you have any idea what could be wrong? Below the settings of my model. Thank you for any help!

Profile photo
Profile photo
3 Replies Latest reply 12 days ago

Li Z.

6:18 AM, Dec 10

Profile photo
Hello for the neural networks course, the instructor says that perceptrons output continuous variables, but I'm pretty sure the node/perceptron only outputs discrete variable (like on/off)?

Profile photo
Profile photo
3 Replies Latest reply 7 days ago

Himanshu M.

Mentor

3:47 AM, Dec 13

Profile photo
Hi everyone !! How are your projects coming along, let me know if you need some help :)

Dhanan J.

7:00 AM, Dec 17

Profile photo
Hello everyone I want to know what is the use of squeeze in plt.imshow()

Profile photo
Profile photo
4 Replies Latest reply about 16 hours ago
