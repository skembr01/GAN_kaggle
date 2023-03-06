## Project and Data

This project involved utilizing a GAN in order to take real photos and change them into the style of a Monet painting. A generative adversarial network (GAN) utilizes a generator and a discriminator. The generator creates photos and the discriminator is then used to judge how well the generated images perform according to some loss function. Then, the generator updates its weights to better itself against the discriminator, while the discriminator also updates its weights to better discriminate against the generator. Thus, in this 'adversarial' way the model improves itself by continuously updating its weighs for its generator to improve against the discriminate and generate images in accordance with some purpose.

In our case, we are generating photos in the style of Monet. The data for this project involved photos of Monet paintings as well as various photographs of everyday objects. The data were provided in two formats. One was in a folder of jpg format and contained 300 photos and 7038 photos. Further, there was a provided format in the form of TFRecords. TFRecords are a storage system and which stores data as set of sequences of binary strings. This type of storage helps in terms of computational memory and performance. This is a specialized storage system which is optimized for TensorFlow. Of these TFRecords, the Monet files, have a length of 5 when read in and a length of 20. Thus, we can already see how the data scale of the data is already reduced by using TFRecords here. This data was loaded into my notebook through tf.io.gfile.glob. 

## Data Preprocessing and Limited EDA

This was my first time working with TFRecords and thus, I utilized TensorFlow documentation on the process of reading in the data as well as a notebook suggested by Kaggle to work on this project. Both sources are listed at the bottom of this page. However, the preprocessing of the data began by setting the format with an image and image name of the TFRecords. Then, with this format, I created a function with python methods to convert data from TFRecord into photo data of size (256, 256, 3). This is done by parsing each example of the TFRecord according to the defined format. Then, it is decoded with 3 channels for RGB photos. Then, it is cast into numeric pixel versions and normalized from a scale to [1, 1]. Next, the images are reshaped into (256, 256, 3) as the initial pictures are of pixel size (256, 256) and as mentioned, we are using a RGB color scale. Lastly, I randomly flipped images in each dataset in efforts to improve model performance. With this function, the data were then loaded into a tf dataset for each batch of Monets and photos.

Next, I wanted to view the first photo in each dataset. This was performed with a function which converts each pixel value onto a scale [0,1] and then utilizes pyplot to show each image.  

## Functions for Modeling

Here, I am defining my functions which will be called in the model itself. First off, in the encoder (or downsampler), we are applying a keras sequential model with a to-be defined filters and size and whether to batch normalize (improving model speed). In this model, we have a convolution layer, followed by a batchnormalizer layer if argument is true, and finally we use a leaky relu layer. Then, we define our decoder (upsampler) which will be used to decode the encoded images in the GAN methodology. For the encoder we again take arguments for number of filters, the size, and here we instead take argument on whether to apply dropout layer (to improve model performance). Here; however, we start with a Convolution transpose layer. Convolution transpose layers are the opposite of convolution layers, which makes sense, since we are decoding the convolution layer of the encoder. Further, we are batch normalizing on every iteration here, and if dropout is applied, I add a dropout rate of 0.7 to improve model speed. Finally, a Relu layer is added. 

## Functions for Modeling Continued (Generator)

Once we have our encoders and decoders defined, we can begin to build a generator. We begin by setting an input layer which will be set for the image size of (256, 256, 3). Then, we will apply a collection of our encoder functions. In sum, we will have eight encoders. The number of filters I chose to use for my first model were:
[64, 128, 128, 128, 128, 256, 256, 256] and I batch normalized all but the first encoder. These were lower than the majority number of filters I saw utilized in the majority of example generators I used. However, I wanted to start here to see how it performed and how it would operate on my personal machine. Then, I set the decoders to use to be 7 decoders and a final Conv2d Transpose layer. The filters chosen for the decoders were [256, 256, 256, 256, 128, 128, 64] and the final layer having 3 filters corresponding to 3 color channels with a tanh activation function. The way the generator operates is it takes an image and it puts it through all of the encoders and then puts it through all of the decoders and eventually returns the entire model in order to be called in the cycle GAN. I visualize the generator function below the function.

## Functions for Modeling (Discriminator)
Next, we define the discriminator. Here we again have an input layer of image shape (256, 256, 3). We then take that image and run it through 3 encoders. In my first model, I utilized 64, 128, and 256 filters for these. After th encoders, I added a zero padding layer to help preserve  image size. Then, I placed it through a convolution layer with 512 filters, a batch normalization layer, a leaky_relu, another zero padding, and a final convolution layer. Again the function ends by returning all of its operations as a model to be called within the cycle GAN. Further, I visualize the operations of the model below the function definition.

After this, I set a specific generator and discriminator for both Monet and photos. Lastly I set each Monet and photo generation and discrimination to have its own optimizer. The optimizer utilized was an Adam operator with a learning rate of 0.002 and a beta_1 of 0.5, based upon observation of similar models in Tensorflow documentation and Kaggle code. 

## Additional Function

In this generate images function I wanted to show what the initial generation looks like for both photos to Monet and Monet to photos. This was by showing each image side by side through pyplot after running through generation for Monet and photos for initial photo and Monet from dataset.

## Loss Functions for Modeling

After defining generators and discriminators, we need to define our loss functions so that model can update its weights and improve itself depending upon loss. First, I created a discriminator loss function which takes as input a real image and a generated image. It then looks at binary cross entropy for both the real image and the generated image. These losses are then added and multiplied by 0.5 and returned. 

Next, for the generator loss, we take binary cross entropy and apply it to the generated image itself. Then, we define a cycle loss which helps improve model consistency. This is done by looking at a cycled image which will generate a Monet from the generated photo and generate a photo from the generated Monet. The cycle loss then measures the loss between the real image whether it be Monet or photo and measures it against the cycle photo or Monet. In this way, we have an additional measure to see how generated images perform against each other by generating from them.

Lastly, we have an identity loss function. This generates a Monet or photo from a Monet or photo respectively. It then measures the loss between the generated image and the input image. Again, we get further loss metrics on how generation is occurring which will help the model fine-tune its weights. 

## The Model

We then build our cycle GAN, largely based upon the defined Cycle GAN's in the keras documentation and the provided notebook by Kaggle. We begin by utilizing the CycleGan class which we set as a keras Model. The class takes as arguments the generators and discriminators defined above and sets them as properties of the class. Then we create a compilation method which takes as input, the optimizers and loss functions defined above. Again, these are set as properties of the class. 

Next, we define a train_step method for the class which takes as input the input data which will be a tuple of a Monet and a photo. In our function we use a GradientTape in accordance with Tensorflow documentation on GANs. We then, generate a Monet, a photo (these from the generators for opposite images), a cycled Monet and photo, and then an identity Monet and photo. We then call our discriminators on the inputted Monet and photo as well as the generated Monet and photo. Once we have called generators and discriminators, we apply each as inputs to our loss functions accordingly for each function. These outputs from the loss functions are then totaled to receive a total loss for generation and discrimination. 

Next, and outside the GenerationTape (again in accordance with Tensorflow documentation) we set the gradients for both generators and discriminators for Monet and photo each. The optimizers properties of the class then get these gradients applied to them to update the optimizer. Lastly, the loss results of generators and discriminators are returned.

After the model creation, the model is called, compiled, and then fitted on the datasets. I initially utilized 10 epochs in my first model iteration. This model received a score of 684.06252 on the Kaggle competition.

## Visualization Function

I then wanted to visualize what the generation of photos in the style of Monet looks like after running through the model. We saw the above at the initial generation and the output image was just grey and unrecognizable as an image. However, here we can make out the images, but we see a bit of the Impressionist side with what looks like a bit of cubism and has a look of the blend of creation and realism with which Monet thrived. 

## Second Iteration of Model

After the initial iteration of my model, I wanted to tune my hyperparameters and check performance. This was done by adding more filters to the generator and the discriminator. Additionally, in the decoder function, I lowered the dropout rate when it was called in the function. Thus, the model should get higher performance, with slower speed. Further, I added 20 epochs in the same vein, which would take longer to model, but should result in better performance.

## Prediction 

This competition was unlike others of this course as we had to create a zip of the images in a Kaggle notebook. Here, we create a images.zip folder and we write the generated photo in style of Monet in each zip. These were saved as 'jpgs'. My work for this was based off https://www.kaggle.com/code/zahid0/gan-tensorflow as I struggled to output the data in my Kaggle notebook.

## Conclusion 

This was my first ever GAN and it was quite a steep learning curve for me, particularly as it has a different structure than other models of this course or which I've used before. However, the GAN here was utilized to generate photos in the style of Monet from provided photos. This occurs by generating photos and seeking for them to beat a discriminator which will look at photos and the generated photos. Hence, the adversarial nature of the GAN. This project provided photos and Monet paintings and our model utilized these as inputs. Our initial model with its hyperparameters received a score of 684 on the Kaggle competition. I decided to apply one more iteration of the model and will run it through the Kaggle notebook. This did not increase my score at all; however, it took approximately 40 minutes to run longer. Given, the nature of this project, I did not have time to proceed further in hyperparameter tuning.

I must say the sources below were invaluable to me given my inexperience with this modeling type. I largely followed the outline provided by the notebook provided by Kaggle as well as the GAN instances found in the Tensorflow examples below.

## Sources: 

https://www.kaggle.com/code/amyjang/monet-cyclegan-tutorial/notebook

https://www.tensorflow.org/tutorials/load_data/tfrecord

https://www.tensorflow.org/tutorials/generative/cyclegan

https://www.kaggle.com/code/zahid0/gan-tensorflow

https://www.tensorflow.org/tutorials/generative/pix2pix#build_the_generator

https://keras.io/examples/generative/cyclegan/

https://medium.com/mostly-ai/tensorflow-records-what-they-are-and-how-to-use-them-c46bc4bbb564
