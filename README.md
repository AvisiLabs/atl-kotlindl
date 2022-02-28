# ATL KotlinDL

This repository contains examples used for our blog and video about KotlinDL.

## Usage

An installed Java environment is required to run Gradle commands. An IDE with good Kotlin support is recommended for viewing and executing the examples.

Each example has its own `main` function that can be used to run that example.

## Included examples

1. [Defining and training a simple convolutional model using MNIST](app/src/main/kotlin/nl/avisi/labs/kotlindl/E1-SimpleModel.kt)
2. [Transfer learning using a ResNet model from the TFModelHub](app/src/main/kotlin/nl/avisi/labs/kotlindl/E2-ModelHubTransferLearning.kt)
3. [Object detection using an SSD model from the ONNXModelHub](app/src/main/kotlin/nl/avisi/labs/kotlindl/E3-ObjectDetectionSSDFromONNX.kt)
4. [Loading a custom model for object detection, in SavedModelBundle format](app/src/main/kotlin/nl/avisi/labs/kotlindl/E4-InferenceUsingCustomSavedModel.kt)
5. [Loading a custom Functional model for image classification, in H5 format](app/src/main/kotlin/nl/avisi/labs/kotlindl/E5-FunctionalH5Model.kt)
6. [Loading a custom Sequential model for image classification, in H5 format](app/src/main/kotlin/nl/avisi/labs/kotlindl/E6-SequentialH5Model.kt)
7. [Landmark detection in an image of a face](app/src/main/kotlin/nl/avisi/labs/kotlindl/E7-FaceLandmarks.kt)
