package nl.avisi.labs.kotlindl

import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.SavedModel
import java.awt.image.BufferedImage
import javax.imageio.ImageIO

object E4Constants {
    const val MODEL_PATH = "./models/custom_model"
    const val EXAMPLE_IMAGE_RESOURCE_PATH = "object-detection-image.jpg"
}

/*
Does not work due to hardcoded types in predict function, cannot be easily changed.
Exception: Expects arg[0] to be uint8 but float is provided
While it is possible to directly use TensorFlow, this is not in scope of these examples

Note: the model is excluded from the repository, but is a FasterRCNN model with ResNet-50 trained using the TensorFlow Object Detection API
 */
fun main() {
    SavedModel.load(E4Constants.MODEL_PATH).use {
        val image = it.loadImage(E4Constants.EXAMPLE_IMAGE_RESOURCE_PATH)
        it.reshape(image.width.toLong(), image.height.toLong(), 3L)
        it.input(Input.PLACEHOLDER)
        it.output(Output.ARGMAX)
        val array = FloatArray(image.width * image.height * 3)
        val floatArray = image.data.getPixels(0, 0, image.width, image.height, array)
        val prediction = it.predict(floatArray, "image_tensor", "detection_classes")
    }
}

private fun SavedModel.loadImage(image: String): BufferedImage {
    val imageURL = this::class.java.classLoader.getResource(image)
        ?: throw IllegalArgumentException("Could not get image $image from resources")

    return ImageIO.read(imageURL)
}
